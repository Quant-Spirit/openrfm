#pragma once
#include "rt_composition.h"
namespace rtc {
struct Dir {RTIndex32 fp,fd,pp,pd; RTIndex ft,pt;};
constexpr int BLOCK_THREADS=256, WARPS=BLOCK_THREADS/32;

struct Candidate {int row,parent,direction,valid,legal,owned,payload,record;};
struct BlockWork {
 long long warp[WARPS],offset[BLOCK_THREADS+1];
 int total,path_size,recent_slot,reversed[66],rows[66],witnesses[66];
 Candidate candidate[BLOCK_THREADS];
};
struct GPU {
 RTGraph g;Meta m;Policy p;Work a;State& s;const int64_t *q,*settings;RTIndex32 fp,fd,pp,pd;BlockWork& block;
};
__device__ inline bool leader(GPU&){return threadIdx.x==0;}
__device__ inline void sync(GPU&){__syncthreads();}
__device__ inline int lane(GPU&){return threadIdx.x;}
__device__ inline int lanes(GPU&){return BLOCK_THREADS;}

__device__ inline int rw_neighbor(GPU& e,const int* path,int n,uint64_t salt){
 int row=path[n-1],best=-1,t=threadIdx.x;uint64_t best_key=~uint64_t(0);
 for(int direction=0;direction<2;++direction){auto ptr=direction?e.pp:e.fp,dst=direction?e.pd:e.fd;
  int begin=__ldg(ptr+row),degree=__ldg(ptr+row+1)-begin;
  int count=rw_window_count(e.p,direction,__ldg(e.fp+row+1)-__ldg(e.fp+row),
                            __ldg(e.pp+row+1)-__ldg(e.pp+row));
  for(int k=t;k<count;k+=BLOCK_THREADS){int j=rw_scan_index(begin,degree,count,k),child=__ldg(dst+j);bool cycle=false;
   for(int i=0;i<n;++i)cycle|=path[i]==child;
   if(cycle||!row_ok(e.g,e.p,e.q,child))continue;
   uint64_t key=rw_draw_key(e,direction,child,salt);
   if(best<0||key<best_key||(key==best_key&&child<best)){best=child;best_key=key;}
  }
 }

 for(int offset=16;offset;offset>>=1){
  auto key=__shfl_down_sync(0xffffffff,best_key,offset);
  int child=__shfl_down_sync(0xffffffff,best,offset);
  if(child>=0&&(best<0||key<best_key||(key==best_key&&child<best))){best=child;best_key=key;}
 }
 if((t&31)==0){e.block.warp[t>>5]=best_key;e.block.offset[t>>5]=best;}
 sync(e);
 if(t==0){best=-1;best_key=~uint64_t(0);
  for(int warp=0;warp<WARPS;++warp){int child=e.block.offset[warp];uint64_t key=e.block.warp[warp];
   if(child>=0&&(best<0||key<best_key||(key==best_key&&child<best))){best=child;best_key=key;}
  }
  e.block.path_size=best;
 }
 sync(e);return e.block.path_size;
}

template<class Integer> __device__ inline Integer scan(GPU& e,Integer value,Integer& total){
 int t=threadIdx.x,l=t&31,w=t>>5;Integer sum=value;
 for(int d=1;d<32;d<<=1){Integer x=__shfl_up_sync(0xffffffff,sum,d);if(l>=d)sum+=x;}
 if(l==31)e.block.warp[w]=sum;sync(e);
 Integer base=0;for(int i=0;i<w;++i)base+=Integer(e.block.warp[i]);
 total=0;for(int i=0;i<WARPS;++i)total+=Integer(e.block.warp[i]);
 sync(e);return base+sum-value;
}
__device__ inline int block_sum(GPU& e,int value){int total;scan(e,value,total);return total;}
__device__ inline int visible_degree(GPU& e,int r,int direction){
 auto ptr=direction?e.pp:e.fp,dst=direction?e.pd:e.fd;
 int begin=__ldg(ptr+r),degree=__ldg(ptr+r+1)-begin;
 int count=e.s.extension?rw_window_count(e.p,direction,
     __ldg(e.fp+r+1)-__ldg(e.fp+r),__ldg(e.pp+r+1)-__ldg(e.pp+r)):degree;
 int visible=0;
 for(int k=threadIdx.x;k<count;k+=BLOCK_THREADS){int j=e.s.extension?
     rw_scan_index(begin,degree,count,k):begin+k;int child=__ldg(dst+j);
  visible+=(j==begin||child!=__ldg(dst+j-1))&&row_ok(e.g,e.p,e.q,child);
 }
 int total=block_sum(e,visible);
 if(e.s.extension&&threadIdx.x==0){e.s.rw_scanned+=count;e.s.rw_truncated+=degree-count;}
 __syncthreads();
 return total;
}

__device__ inline void expand(GPU& e,bool ego){
 auto& s=e.s;auto a=e.a;auto p=e.p;int t=threadIdx.x;
 for(int base=0;base<(ego?s.ego_rows:s.np)&&!s.error&&(ego||!s.frontier_capped);){
  int n=minimum(BLOCK_THREADS,(ego?s.ego_rows:s.np)-base);long long edges=0;
  if(t<n){auto v=a.plan+PW*(base+t);int r=v[0];
   if(v[3]<(ego?p.seed_hops:p.h))edges=int64_t(__ldg(e.fp+r+1)-__ldg(e.fp+r))+(__ldg(e.pp+r+1)-__ldg(e.pp+r));
  }

  long long total=0,offset=scan(e,edges,total);e.block.offset[t]=offset;
  if(t==0)e.block.offset[BLOCK_THREADS]=total;sync(e);
  for(long long start=0;start<total&&!s.error&&(ego||!s.frontier_capped);start+=BLOCK_THREADS){
   Candidate c{};long long flat=start+t;
   if(flat<total){int lo=0,hi=n;
    while(lo<hi){int mid=(lo+hi)/2;if(e.block.offset[mid+1]<=flat)lo=mid+1;else hi=mid;}
    c.parent=base+lo;auto v=a.plan+PW*c.parent;int r=v[0];long long local=flat-e.block.offset[lo];
    int fbegin=__ldg(e.fp+r),fcount=__ldg(e.fp+r+1)-fbegin;
    c.direction=local>=fcount;auto ptr=c.direction?e.pp:e.fp,dst=c.direction?e.pd:e.fd;
    int begin=__ldg(ptr+r),offset=c.direction?local-fcount:local;
    int j=p.canonical?__ldg(ptr+r+1)-1-offset:begin+offset;c.row=__ldg(dst+j);
    c.valid=j==begin||c.row!=__ldg(dst+j-1);
    if(c.valid&&!ego){
     for(int i=c.parent;i>=0;i=a.plan[PW*i+1])if(a.plan[PW*i]==c.row){c.valid=0;break;}
     c.valid=c.valid&&structural(e.g,p,e.q,c.row);
     if(p.canonical)c.valid=c.valid&&row_ok(e.g,p,e.q,c.row);
    }
    if(c.valid&&!ego){c.record=stream_record(e,v[2],c.direction,e.g.table[c.row]);c.valid=c.record>=0;}
    if(c.valid){c.legal=row_ok(e.g,p,e.q,c.row);

     if(!ego&&c.legal&&s.np<s.limit){int parent_sid=v[2],direction=c.direction;int64_t table=e.g.table[c.row];
      if(parent_sid>=0){auto branch=a.scratch+SW*a.scratch[SW*parent_sid+12];direction=branch[2];table=branch[3];}
      c.owned=candidate_owned(e,c.row,direction,table);c.payload=candidate_payload(e,c.row);
     }
    }
   }
   e.block.candidate[t]=c;sync(e);
   if(t==0)for(int i=0;i<int((total-start)<BLOCK_THREADS?(total-start):BLOCK_THREADS)&&!s.error&&(ego||!s.frontier_capped);++i){auto c=e.block.candidate[i];if(!c.valid)continue;
    int depth=a.plan[PW*c.parent+3];
    if(ego){int h=slot(c.row,a.seen,p.H);if(a.seen[h]!=-1||!c.legal)continue;
     if(s.ego_rows==p.seed_rows){++s.ego_cap;continue;}
     a.seen[h]=c.row;a.plan[PW*s.ego_rows]=c.row;a.plan[PW*s.ego_rows+3]=depth+1;++s.ego_rows;
    }else{
     int sid=stream_prepared(e,a.plan[PW*c.parent+2],c.direction,e.g.table[c.row],c.record);
     if(s.error)break;if(sid>=0)add_prepared(e,c.row,c.parent,sid,depth+1,c.owned,c.payload,c.legal);
    }
   }
   sync(e);
  }
  base+=n;
 }
}
__device__ inline void seed_ego(GPU& e){
 if(leader(e)){int root=e.q[0];e.a.plan[0]=root;e.a.plan[3]=0;e.s.ego_rows=1;e.a.seen[slot(root,e.a.seen,e.p.H)]=root;}
 sync(e);expand(e,true);
 for(int i=threadIdx.x;i<e.s.ego_rows;i+=BLOCK_THREADS)e.a.plan[PW*i+1]=slot(e.a.plan[PW*i],e.a.seen,e.p.H);
 sync(e);
 for(int i=threadIdx.x;i<e.s.ego_rows;i+=BLOCK_THREADS)e.a.seen[e.a.plan[PW*i+1]]=-1;
 sync(e);
}
__device__ inline void discover(GPU& e){
 if(leader(e)){auto v=e.a.plan;v[0]=e.q[0];v[1]=-1;v[2]=-1;v[3]=0;v[4]=-1;v[5]=0;v[6]=0;e.s.np=1;}
 sync(e);expand(e,false);if(e.s.error)return;

 if(e.p.reserve){if(leader(e)){e.s.limit=e.s.frontier;e.s.frontier_capped=0;}sync(e);}
 int pool=-1;int pool_size=e.m.pool_size<0?e.p.N:e.m.pool_size;
 for(int64_t base=0;base<pool_size&&!e.s.error&&!e.s.frontier_capped;base+=BLOCK_THREADS){
  int64_t k=base+threadIdx.x,i=e.p.reserve?pool_size-1-k:k;
  int r=k<pool_size?(e.m.pool_size<0?int(i):int(e.m.pool[i])):0;Candidate c{};c.row=r;
  c.valid=k<pool_size&&pool_ok(e,r);
  if(c.valid){c.owned=candidate_owned(e,r,2,e.g.table[e.q[0]]);c.payload=candidate_payload(e,r);}
  e.block.candidate[threadIdx.x]=c;sync(e);
  if(leader(e))for(int i=0;i<int((pool_size-base)<BLOCK_THREADS?(pool_size-base):BLOCK_THREADS)&&!e.s.error&&!e.s.frontier_capped;++i){auto c=e.block.candidate[i];
   if(!c.valid)continue;if(pool<0){pool=stream(e,-1,2,e.g.table[e.q[0]]);if(pool>=0)e.s.pool_sid=pool+1;}
   if(e.s.error)break;if(pool>=0)add_prepared(e,c.row,0,pool,e.p.h,c.owned,c.payload,true);
  }
  sync(e);
 }
}

__device__ inline bool candidate_before(GPU& e,int i,int j){
 int row=e.a.plan[PW*i],other=e.a.plan[PW*j];
 uint64_t key=e.a.seen[2*e.p.F+i],other_key=e.a.seen[2*e.p.F+j];

 return key<other_key||(key==other_key&&(row<other||(row==other&&i<j)));
}
__device__ inline void prepare_streams(GPU& e){
 auto a=e.a;int t=threadIdx.x;
 for(int i=t;i<e.s.np;i+=BLOCK_THREADS)a.seen[2*e.p.F+i]=rt_mix(uint64_t(e.q[3])^a.plan[PW*i]);

 if(!e.p.weight_policy){for(int i=t;i<e.s.ns;i+=BLOCK_THREADS)a.scratch[SW*i+11]=isqrt(uint64_t(a.scratch[SW*i+4])*(1<<20));}
 else{auto records=a.schema+e.s.schema_offset;
  for(int rec=t;rec<e.s.schema_size;rec+=BLOCK_THREADS){auto v=records+SCHEMA_WIDTH*rec;int sid=int(a.mapping[v[3]]);
   if(sid>=0)a.scratch[SW*sid+11]=isqrt(uint64_t(a.scratch[SW*sid+4])*uint64_t(v[5]));}}
 sync(e);

 if(t==0){int offset=0;

  for(int i=0;i<e.s.ns;++i){auto v=a.scratch+SW*i;
   a.scratch[SW*v[12]+16]+=e.p.weight_policy?isqrt(uint64_t(v[4])*(1<<20)):v[11];v[15]=offset;offset+=v[4];}
  e.block.total=offset;signature_order(e);
 }
 sync(e);int count=e.block.total;

 for(int i=t;i<e.s.ns;i+=BLOCK_THREADS){auto v=a.scratch+SW*i;int offset=v[15];
  for(int j=v[13];j>=0;j=a.plan[PW*j+4])a.seen[offset++]=j;
  v[13]=v[15];v[14]=offset;
 }
 sync(e);

 auto input=a.seen,output=a.seen+e.p.F;
 for(int width=1;width<count;width*=2){
  for(int i=t;i<count;i+=BLOCK_THREADS){int witness=input[i],sid=a.plan[PW*witness+2];auto v=a.scratch+SW*sid;
   int begin=v[15],end=v[14],local=i-begin,run=local/width,partner=begin+(run^1)*width;
   int lo=minimum(partner,end),hi=minimum(partner+width,end),first=lo;
   while(lo<hi){int mid=(lo+hi)/2;if(candidate_before(e,input[mid],witness))lo=mid+1;else hi=mid;}
   int dest=begin+(run/2)*(2*width)+local%width+(lo-first);output[dest]=witness;
  }
  sync(e);auto tmp=input;input=output;output=tmp;
 }
 if(input!=a.seen){for(int i=t;i<count;i+=BLOCK_THREADS)a.seen[i]=input[i];sync(e);}
}
__device__ inline bool extension_slot_before(GPU& e,int i,int j){
 if(i<0)return false;if(j<0)return true;
 int row=e.a.plan[PW*e.a.seen[i]],other=e.a.plan[PW*e.a.seen[j]];
 return extension_before(e,row,other)||(row==other&&i<j);
}
__device__ inline bool recent_slot_before(GPU& e,int i,int j){
 if(i<0)return false;if(j<0)return true;
 int row=e.a.plan[PW*e.a.seen[i]],other=e.a.plan[PW*e.a.seen[j]];
 return recent_before(e,row,other)||(row==other&&i<j);
}

__device__ inline int recent_candidate(GPU& e,int begin,int end){
 int best=-1,t=threadIdx.x;
 for(int i=begin+t;i<end;i+=BLOCK_THREADS)if(recent_slot_before(e,i,best))best=i;
 for(int d=16;d;d>>=1){int other=__shfl_down_sync(0xffffffff,best,d);
  if((t&31)+d<32&&recent_slot_before(e,other,best))best=other;
 }
 if((t&31)==0)e.block.warp[t>>5]=best;sync(e);
 best=-1;for(int i=0;i<WARPS;++i)if(recent_slot_before(e,e.block.warp[i],best))best=e.block.warp[i];
 sync(e);
 if(t==0)e.block.recent_slot=best;
 sync(e);return best<0?-1:int(e.a.seen[best]);
}
__device__ inline int best_candidate(GPU& e,int sid){
 auto v=e.a.scratch+SW*sid;int begin=v[13],end=v[14];
 if(!e.s.extension&&(e.s.recent||e.p.canonical))return recent_candidate(e,begin,end);
 if(!e.s.extension)return begin<end?int(e.a.seen[begin]):-1;
 int best=-1,t=threadIdx.x;
 for(int i=begin+t;i<end;i+=BLOCK_THREADS)if(extension_slot_before(e,i,best))best=i;
 for(int d=16;d;d>>=1){int other=__shfl_down_sync(0xffffffff,best,d);
  if((t&31)+d<32&&extension_slot_before(e,other,best))best=other;
 }
 if((t&31)==0)e.block.warp[t>>5]=best;sync(e);
 best=-1;for(int i=0;i<WARPS;++i)if(extension_slot_before(e,e.block.warp[i],best))best=e.block.warp[i];
 sync(e);
 if(t==0&&best>=0){auto saved=e.a.seen[begin];e.a.seen[begin]=e.a.seen[best];e.a.seen[best]=saved;}
 sync(e);return begin<end?int(e.a.seen[begin]):-1;
}
__device__ inline void consume_candidate(GPU& e,int sid){++e.a.scratch[SW*sid+13];}
__device__ inline int choose_stream(GPU& e){
 int best=-1,t=threadIdx.x;
 for(int i=t;i<e.s.ns;i+=BLOCK_THREADS)if(e.a.scratch[SW*i+6]+e.a.scratch[SW*i+9]<e.a.scratch[SW*i+4]&&stream_before(e,i,best))best=i;
 for(int d=16;d;d>>=1){int other=__shfl_down_sync(0xffffffff,best,d);if((t&31)+d<32&&stream_before(e,other,best))best=other;}
 if((t&31)==0)e.block.warp[t>>5]=best;sync(e);
 best=-1;for(int i=0;i<WARPS;++i)if(stream_before(e,e.block.warp[i],best))best=e.block.warp[i];
 sync(e);return best;
}

__device__ inline int identity_witness(GPU& e){
 int best=INT_MAX,t=threadIdx.x;
 if(e.p.provenance&&e.p.forecast)for(int i=t+1;i<e.s.np;i+=BLOCK_THREADS){auto v=e.a.plan+PW*i;
  if(v[0]==e.m.entity[e.q[0]]&&v[1]==0&&e.a.scratch[SW*v[2]+2]==0)best=minimum(best,i);
 }
 for(int d=16;d;d>>=1)best=minimum(best,__shfl_down_sync(0xffffffff,best,d));
 if((t&31)==0)e.block.warp[t>>5]=best;sync(e);
 for(int i=0;i<WARPS;++i)best=minimum(best,int(e.block.warp[i]));
 sync(e);return best==INT_MAX?-1:best;
}
__device__ inline bool discovery_capped(GPU& e){
 int capped=0;for(int i=threadIdx.x;i<e.s.ns;i+=BLOCK_THREADS)capped|=e.a.scratch[SW*i+8]>0;
 return block_sum(e,capped)>0||e.s.pruned>0;
}
__device__ inline void receipt_totals(GPU& e){
 for(int field=4;field<=8;++field){long long value=0,total=0;
  for(int i=threadIdx.x;i<e.s.ns;i+=BLOCK_THREADS)value+=e.a.scratch[SW*i+field];

  scan(e,value,total);
  if(leader(e)){if(field<8)e.a.receipt[19+field]=total;
   else e.a.receipt[27]=total>INT64_MAX-e.s.pruned?INT64_MAX:e.s.pruned+total;
  }
 }
 sync(e);
}

__device__ inline int admit(GPU& e,const int* rows,int n,bool filler=false,const int* witnesses=nullptr){
 auto a=e.a;auto p=e.p;auto g=e.g;auto m=e.m;auto q=e.q;auto& s=e.s;int t=threadIdx.x;
 int needed=0,kenc=s.kenc;

 for(int i=0;i<n;++i){int r=rows[i],features=0,labels=0;
  for(int64_t c=g.cp[r]+t;c<g.cp[r+1];c+=BLOCK_THREADS){
   if(!payload_cell(e,r,c,filler)||a.emitted[slot(c,a.emitted,p.E)]!=-1)continue;
   bool label=r!=q[0]&&c==m.target[r];features+=!label;labels+=label;
  }
  features=block_sum(e,features);labels=minimum(block_sum(e,labels),maximum(0,p.kmax-kenc));
  needed+=features+labels;kenc+=labels;
 }
 if(s.count+needed>e.settings[0])return -1;
 int end=s.count+needed;
 for(int i=0;i<n;++i){int r=rows[i],role=filler?4:annotation(g,m,q,r);
  for(int64_t base=g.cp[r];base<g.cp[r+1];base+=BLOCK_THREADS){int64_t c=base+t;
   bool label=r!=q[0]&&c==m.target[r];
   int keep=c<g.cp[r+1]&&payload_cell(e,r,c,filler)&&a.emitted[slot(c,a.emitted,p.E)]==-1&&(!label||s.kenc<p.kmax);
   int count=0,offset=scan(e,keep,count);if(!count)continue;
   int rh=row_slot(r,a.row_seen,p.E),start=boundary_offset(p)+MAX_SEQ_LEN;
   bool new_row=a.row_seen[rh]==-1;sync(e);
   if(new_row){
    int64_t packed=p.provenance?provenance(e,r,witnesses?witnesses[i]:-1,filler):0;
    if(t==0){a.row_seen[rh]=(int64_t(r)<<32)|s.rows;if(p.provenance)a.receipt[start+s.rows]=packed;++s.rows;}
    sync(e);
   }
   int position=s.count+offset;
   if(keep){a.cells[position]=c;a.roles[position]=role;a.receipt[boundary_offset(p)+position]=end;
    if(p.provenance)a.receipt[start+MAX_SEQ_LEN+position]=a.row_seen[rh]&0xffffffffULL;
   }

   e.block.candidate[t].valid=keep;sync(e);
   int labels=block_sum(e,keep&&label),targets=block_sum(e,keep&&r==q[0]&&c==m.target[r]);
   int texts=block_sum(e,keep&&m.types[c]==1);
   if(t==0){for(int j=0;j<int((g.cp[r+1]-base)<BLOCK_THREADS?(g.cp[r+1]-base):BLOCK_THREADS);++j)
     if(e.block.candidate[j].valid)a.emitted[slot(base+j,a.emitted,p.E)]=base+j;
    s.count+=count;s.kenc+=labels;s.targets+=targets;s.text+=texts;s.filler+=role==4?count:0;s.own_cells+=role==1?count:0;
    s.peers+=role==2?labels:0;s.history+=role==1?labels:0;
    if(s.extension)for(int j=0;j<int((g.cp[r+1]-base)<BLOCK_THREADS?(g.cp[r+1]-base):BLOCK_THREADS);++j)
      if(e.block.candidate[j].valid)a.table_counts[extension_group(e,r)]++;
   }
   sync(e);
  }
 }
 return needed;
}

__device__ inline bool take(GPU& e,int sid){
 int best=best_candidate(e,sid);if(best<0)return true;
 auto& b=e.block;
 if(leader(e)){b.path_size=0;
  for(int i=best;i>=0;i=e.a.plan[PW*i+1]){
   if(b.path_size==66){e.s.error=8;break;}b.reversed[b.path_size++]=i;
  }
  for(int i=0;i<b.path_size;++i){b.witnesses[i]=b.reversed[b.path_size-i-1];b.rows[i]=e.a.plan[PW*b.witnesses[i]];}
 }
 sync(e);if(e.s.error)return false;
 int charged=admit(e,b.rows,b.path_size,false,b.witnesses);
 if(charged<0){if(e.s.extension){if(leader(e)){e.a.plan[PW*best+5]=1;++e.a.scratch[SW*sid+9];consume_candidate(e,sid);}sync(e);}return false;}
 if(leader(e)){auto v=e.a.scratch+SW*sid;e.a.plan[PW*best+5]=1;++v[6];v[10]+=charged;
  if((e.s.recent||e.p.canonical)&&!e.s.extension){int begin=v[13],at=e.block.recent_slot;auto chosen=e.a.seen[at];
   for(int j=at;j>begin;--j)e.a.seen[j]=e.a.seen[j-1];e.a.seen[begin]=chosen;}
  consume_candidate(e,sid);}
 sync(e);return true;
}
__global__ void initialize(Policy p,int64_t* visits,int64_t* candidates,int64_t* scores,int64_t* cells,int64_t* roles,int64_t* emitted,int64_t* row_seen){
 int b=blockIdx.x,t=threadIdx.x;
 for(int i=t;i<p.W*p.K;i+=256)visits[b*p.W*p.K+i]=-1;
 for(int i=t;i<p.C;i+=256){candidates[b*p.C+i]=-1;scores[b*p.C+i]=0;}
 for(int i=t;i<p.L;i+=256){cells[b*p.L+i]=-1;roles[b*p.L+i]=0;}
 for(int i=t;i<p.E;i+=256){emitted[b*p.E+i]=-1;row_seen[b*p.E+i]=-1;}
}
__global__ void composition(RTGraph g,Dir d,Meta m,Policy p,const int64_t* q,const int64_t* settings,
 int64_t* visits,int64_t* cells,int64_t* roles,int64_t* receipt,int64_t* plan,int64_t* tables,int64_t* seen,int64_t* emitted,int64_t* rows,int64_t* scratch,int64_t* candidates,int64_t* scores,const int64_t* schema,int64_t* mapping,int64_t* seed_work){
 int b=blockIdx.x,t=threadIdx.x;
 __shared__ State state;
 __shared__ BlockWork block;
 if(t==0)state=State{};
 for(int i=t;i<receipt_size(p);i+=blockDim.x)receipt[b*receipt_size(p)+i]=i>=STREAM_OFFSET?-1:0;
 for(int i=t;i<p.H;i+=blockDim.x)seen[b*p.H+i]=-1;
 for(int i=t;i<p.S;i+=blockDim.x)mapping[b*p.S+i]=-1;
 __syncthreads();

 int64_t core_total=p.core?minimum(settings[b*5],int64_t(p.core)):settings[b*5];
 int64_t chosen[5]={core_total,settings[b*5+1],settings[b*5+2],settings[b*5+3],settings[b*5+4]};
 Work a{cells+b*p.L,roles+b*p.L,receipt+b*receipt_size(p),plan+b*p.F*PW,tables+b*group_capacity(p),seen+b*p.H,emitted+b*p.E,rows+b*p.E,scratch+b*p.S*SW,candidates+b*p.C,scores+b*p.C,schema,mapping+b*p.S,seed_work+b*seed_work_size(p)};
 GPU engine{g,m,p,a,state,q+b*5,chosen,d.fp,d.fd,d.pp,d.pd,block};
 int64_t* walk=visits+b*p.W*p.K;

 if(!general_tiers(p))run(engine);
 else if(run_head(engine)){
  int scale=core_total/1024>1?int(core_total/1024):1;
  RWSerial serial{g,m,p,a,state,q+b*5,chosen,d.fp,d.fd,d.pp,d.pd};
  if(p.own_recent){if(t==0)own_recent_tier(serial);__syncthreads();}
  else if(p.own_labels){if(t==0)own_label_tier(serial,scale);__syncthreads();}else own_history(engine);
  if(quota_tiers(p)){
   int64_t share=core_total*p.quota_share/100;
   chosen[0]=p.quota_share?(state.count>share?int64_t(state.count):share):core_total;
   quota_tier(engine,scale);chosen[0]=core_total;
  }
  if(p.seed_percent){
   int64_t cap=state.count+core_total*p.seed_percent/100;
   chosen[0]=cap<core_total?cap:core_total;
   if(t==0)seed_tier(serial,walk);
   __syncthreads();chosen[0]=core_total;
  }
  if(t==0)state.blocked=p.stream_fill?0:1;
  __syncthreads();
  if(p.stream_fill)run_tail(engine);
 }
 if(p.core&&settings[b*5]>p.core&&!state.error){
  chosen[0]=settings[b*5];
  begin_extension(engine);
  if(p.mixed&&p.peer_ext){
   chosen[0]=state.count+(settings[b*5]-p.core)*p.peer_ext/100;
   if(t==0){RWSerial px{g,m,p,a,state,q+b*5,chosen,d.fp,d.fd,d.pp,d.pd};peer_extension(px);}
   __syncthreads();chosen[0]=settings[b*5];
  }
  if(p.mixed)chosen[0]=state.count+(settings[b*5]-p.core)*p.rw_percent/100;
  if(t==0){
   RWSerial rw{g,m,p,a,state,q+b*5,chosen,d.fp,d.fd,d.pp,d.pd};
   if(!p.mixed||p.rw_percent){
    if(state.walked!=1){if(state.walked==2)for(int i=0;i<p.W*p.K;++i)walk[i]=-1;generate_rw(rw,walk);}
    extend_rw(rw,walk);
   }
  }
  __syncthreads();
  if(p.mixed){chosen[0]=settings[b*5];extend_paths(engine);}
 }
 finish(engine);
}
}
Tensors rt_compose_v2_cuda(Tensors g,Tensors d,Tensors m,Tensor q,Tensor settings,Tensors w,std::vector<int64_t> v){
 c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream();int B=q.size(0);
 rtc::Policy p{int(v[0]),int(v[1]),int(v[2]),int(v[3]),int(v[4]),int(v[5]),int(v[6]),int(v[7]),int(v[8]),int(v[9]),int(v[10]),int(v[11]),int(v[12]),int(v[13]),int(v[14]),int(v[15]),int(v[16]),int(v[17]),int(v[18]),int(v[19]),int(v[20]),int(v[21]),int(v[22]),int(v[23]),int(v[24]),int(v[25]),int(v[26]),int(v[27]),int(v[28]),int(v[29]),int(v[30]),int(v[31]),int(v[32]),int(v[33]),int(v[34]),int(v[35]),int(v[36]),int(v[37]),int(v[38]),int(v[39]),int(v[40]),int(v[41]),int(v[42]),int(v[43]),int(v[44]),int(v[45]),int(v[46]),int(v[47]),int(v[48]),int(v[49]),int(v[50]),int(v[51]),int(v[52]),int(v[53]),int(v[54]),int(v[55])};
 RTGraph graph{rt_index(g[0]),rt_index(g[1]),rt_index(g[2]),rt_index(g[3]),rt_index(g[4]),rt_index(g[5]),rt_index(g[6]),rt_index(g[7]),rt_index(g[8]),rt_index(g[9]),rt_index(g[10])};
 rtc::Dir dir{rt_address(d[0]),rt_address(d[1]),rt_address(d[3]),rt_address(d[4]),rt_index(d[2]),rt_index(d[5])};
 rtc::Meta meta{rt_index(m[0]),rt_index(m[1]),rt_index(m[2]),rt_index(m[3]),rt_index(m[4]),rt_index(m[5]),rt_index(m[6]),rt_index(m[7]),rt_index(m[8]),rt_index(m[9]),rt_index(m[10]),rt_index(m[11]),rt_index(m[12]),rt_index(m[13]),rt_index(m[14]),rt_index(m[15]),rt_index(m[16]),rt_index(m[17]),rt_index(m[18]),rt_address(d[0]),rt_address(d[1]),rt_index((m.size()>=20&&m[19].numel()>0)?m[19]:g[4]),m.size()>=20&&m[19].numel()>0};
 if(m.size()>=21){meta.pool=rt_index(m[20]);meta.pool_size=int(m[20].numel());}
 if(m.size()==22)meta.recent=rt_index(m[21]);

 auto candidates=w[1].data_ptr<int64_t>(),scores=w[2].data_ptr<int64_t>();
 w[0].fill_(-1);
 if(B){
  rtc::initialize<<<B,256,0,stream>>>(p,w[0].data_ptr<int64_t>(),candidates,scores,w[3].data_ptr<int64_t>(),w[4].data_ptr<int64_t>(),w[9].data_ptr<int64_t>(),w[10].data_ptr<int64_t>());C10_CUDA_KERNEL_LAUNCH_CHECK();
  rtc::composition<<<B,256,0,stream>>>(graph,dir,meta,p,q.data_ptr<int64_t>(),settings.data_ptr<int64_t>(),w[0].data_ptr<int64_t>(),w[3].data_ptr<int64_t>(),w[4].data_ptr<int64_t>(),w[5].data_ptr<int64_t>(),w[6].data_ptr<int64_t>(),w[7].data_ptr<int64_t>(),w[8].data_ptr<int64_t>(),w[9].data_ptr<int64_t>(),w[10].data_ptr<int64_t>(),w[11].data_ptr<int64_t>(),candidates,scores,w[12].data_ptr<int64_t>(),w[13].data_ptr<int64_t>(),w[14].data_ptr<int64_t>());C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 return {w[0],w[1],w[2],w[3],w[4],w[5]};
}
