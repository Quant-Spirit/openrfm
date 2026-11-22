#pragma once
#include "rt_retrieval.h"
#ifdef __CUDACC__
#define RTC_RUN __device__
#else
#define RTC_RUN
#endif
namespace rtc {

constexpr int R=19, PRIORITY_R=22, SW=20, PW=7;
constexpr int STREAM_OFFSET=36, STREAM_WIDTH=13, SCHEMA_HEADER=6, SCHEMA_WIDTH=6;
constexpr int MAX_SEQ_LEN=16384;
constexpr int OWN_HISTORY_ROWS=128;
struct Meta {
 RTIndex types,card,target,classes,entity,hp,hr,caps,gp,ge,gt,gd,time,kp,keys;
 RTIndex table_ids,root_capacity,stream_mask,task_capacity;
 RTIndex32 fp,fd;
 RTIndex source_time;
 bool source_rank;

 RTIndex pool{nullptr,8}; int pool_size=-1;

 RTIndex recent{nullptr,8};
};

struct Policy {int W,K,C,L,F,H,E,T,history,kmax,h,text,card,forecast,anchor,rate,strata,filler,probes,truncate,N,S,seed_hops,seed_rows,provenance,core,mixed,rw_percent,time_unit,rw_scan_budget,scale_discovery,weight_policy,reserve,seed_percent,seed_local,seed_balance,quota,quota_member,quota_share,own_labels,seed_rank,seed_fill,seed_hide,seed_local_policy,seed_fanout,stream_fill,quota_parent,quota_grandparent,quota_grandchild,quota_order,seed_hop_cap,peer_ext,peer_ext_rows,seed_program_mature,own_recent,canonical;};
constexpr int POLICY_SIZE=56;
static_assert(sizeof(Policy)==POLICY_SIZE*sizeof(int),"Policy fields must match the POLICY_SIZE vector entries");
EP_HD inline int group_capacity(Policy p){return p.T*(p.mixed?p.strata+1:1);}
EP_HD inline int boundary_offset(Policy p){return STREAM_OFFSET+p.S*STREAM_WIDTH;}
EP_HD inline int receipt_size(Policy p){return boundary_offset(p)+(p.provenance?3:1)*MAX_SEQ_LEN;}

constexpr int PROGRAM_PATHS=16;
EP_HD inline int seed_work_size(Policy p){return p.seed_rank==2&&p.seed_percent?(5+PROGRAM_PATHS)*p.seed_hop_cap:1;}

struct State {
 int count,rows,kenc,targets,text,filler,peers,history,np,error;
 int ns,seed_end,own_cells,blocked,hash_entries,extension;
 int rw_scanned,rw_truncated,rw_graph_exhausted,rw_last_truncated;
 int schema_offset,schema_size,saturated,ego_rows,frontier_capped;
 int frontier;

 int limit,walked,pool_sid,recent;
 int64_t total,pruned,ego_cap;
};
struct Work {
 int64_t *cells,*roles,*receipt,*plan,*table_counts,*seen,*emitted,*row_seen,*scratch,*candidates,*scores;
 const int64_t *schema;
 int64_t *mapping;
 int64_t *seed_work;
};
EP_HD inline int minimum(int a,int b){return a<b?a:b;}
EP_HD inline int maximum(int a,int b){return a>b?a:b;}

EP_HD inline int frontier_bound(Policy p,int capacity,int64_t natural){
 if(!p.scale_discovery)return capacity;
 return minimum(capacity,maximum(p.seed_rows,int(int64_t(capacity)*natural/MAX_SEQ_LEN)));
}
EP_HD inline bool time_ok(int64_t t,int64_t cut,bool strict){return t==-(int64_t(1)<<62)||rt_time_ok(t,cut,strict);}
EP_HD inline bool policy_time_ok(Policy p,int64_t t,int64_t cut,bool strict){return p.history==2||time_ok(t,cut,strict);}
EP_HD inline bool structural(RTGraph g,Policy p,const int64_t* q,int r){
 return (p.history!=3||g.task[r]==0)&&rt_row_ok_forecast(g,r,q[0],p.forecast);
}
EP_HD inline bool row_ok(RTGraph g,Policy p,const int64_t* q,int r){
 return structural(g,p,q,r)&&(r==q[0]||policy_time_ok(p,g.rowtime[r],q[1],q[4]));}
EP_HD inline int slot(int64_t key,int64_t* hash,int size){int i=rt_mix(key)&(size-1);while(hash[i]!=-1&&hash[i]!=key)i=(i+1)&(size-1);return i;}
EP_HD inline bool drop_cell(RTGraph g,Meta,const int64_t* q,Policy p,int row,int64_t c){return g.drop[c]&&(p.history!=2||row==q[0]);}
EP_HD inline bool label_ok(RTGraph g,Meta m,const int64_t* q,Policy p,int r){int64_t c=m.target[r];
 return r!=q[0]&&row_ok(g,p,q,r)&&g.table[r]==g.table[q[0]]&&c>=0&&
  ((p.history==2&&!drop_cell(g,m,q,p,r,c))||(!g.drop[c]&&(p.history==1||p.history==3)&&g.avail[c]<q[1]))&&m.classes[r]>=0;}
EP_HD inline bool feature_ok(RTGraph g,Meta m,const int64_t* q,Policy p,int row,int64_t c){
 if(g.label[c]||g.col[c]==q[2]||drop_cell(g,m,q,p,row,c)||!row_ok(g,p,q,row)||(p.history!=2&&g.avail[c]>q[1]))return false;
 int t=m.types[c];return row==q[0]||p.text==0||(p.text==1?(t==0||t==2||t==3):(t!=1||(m.card[c]>=0&&m.card[c]<=p.card)));}
EP_HD inline int isqrt(uint64_t x){uint64_t lo=0,hi=uint64_t(1)<<31;while(lo+1<hi){auto mid=(lo+hi)/2;if(mid*mid<=x)lo=mid;else hi=mid;}return int(lo);}
EP_HD inline int annotation(RTGraph g,Meta m,const int64_t* q,int r){
 if(r==q[0])return 0;
 int root_keys=m.kp[q[0]+1]-m.kp[q[0]];
 if(root_keys>1||(root_keys==1&&m.keys[m.kp[q[0]]]==q[0])){
  for(auto i=m.kp[q[0]];i<m.kp[q[0]+1];++i)
   for(auto j=m.kp[r];j<m.kp[r+1];++j)if(m.keys[i]==m.keys[j])return 1;
  return g.table[r]==g.table[q[0]]?2:3;
 }
 int root=q[0],key=m.hp[root+1]>m.hp[root]||m.entity[root]<0?root:m.entity[root];
 if(m.entity[r]>=0&&m.entity[r]==key)return 1;
 return g.table[r]==g.table[root]?2:3;
}
EP_HD inline uint64_t signature(int64_t parent,int direction,int64_t table){
 return rt_mix(uint64_t(parent)^rt_mix(direction+1)^rt_mix(table+17))&0x7fffffffffffffffULL;
}
EP_HD inline int rw_window_count(Policy p,int direction,int forward_degree,int reverse_degree){
 int first=minimum(forward_degree,(p.rw_scan_budget+1)/2);
 int second=minimum(reverse_degree,p.rw_scan_budget-first);
 return direction==0?first:second;
}
EP_HD inline int rw_scan_index(int begin,int degree,int count,int index){

 return begin+int((uint64_t(2)*uint64_t(index)+1)*uint64_t(degree)
                  /(uint64_t(2)*uint64_t(count)));
}

EP_HD inline int log_bucket(uint64_t n){int b=0;for(n+=1;n>1;n>>=1)++b;return b;}
EP_HD inline int delta_bucket(int64_t t,int64_t cut,bool query){
 if(query)return 0;if(t==-(int64_t(1)<<62)||cut==-(int64_t(1)<<62))return -128;

 uint64_t delta=t>=cut?uint64_t(t)-uint64_t(cut):uint64_t(cut)-uint64_t(t);
 int b=delta==~uint64_t(0)?63:minimum(63,log_bucket(delta));return t>=cut?b:-b;
}

template<class Sink> RTC_RUN int extension_bucket(Sink& e,int row){
 int64_t t=e.g.rowtime[row],cut=e.q[1];
 if(t==-(int64_t(1)<<62)||cut==-(int64_t(1)<<62))return e.p.strata;
 uint64_t delta=t>=cut?uint64_t(t)-uint64_t(cut):uint64_t(cut)-uint64_t(t);
 uint64_t n=delta/e.p.time_unit;int bucket=0;
 while(n&&bucket<e.p.strata-1){++bucket;n>>=2;}
 return bucket;
}
template<class Sink> RTC_RUN int extension_group(Sink& e,int row){
 int lo=0,hi=e.p.T;auto table=e.g.table[row];
 while(lo<hi){int mid=(lo+hi)/2;if(e.m.table_ids[mid]<table)lo=mid+1;else hi=mid;}
 return lo*(e.p.strata+1)+extension_bucket(e,row);
}
template<class Sink> RTC_RUN bool extension_before(Sink& e,int row,int other){
 if(row<0)return false;if(other<0)return true;
 auto cost=e.a.table_counts[extension_group(e,row)],old=e.a.table_counts[extension_group(e,other)];
 if(cost!=old)return cost<old;
 int age=extension_bucket(e,row),old_age=extension_bucket(e,other);
 if(age!=old_age)return age<old_age;
 uint64_t key=rt_mix(uint64_t(e.q[3])^row),old_key=rt_mix(uint64_t(e.q[3])^other);
 return key<old_key||(key==old_key&&row<other);
}
template<class Sink> RTC_RUN uint64_t rw_draw_key(Sink& e,int direction,int child,uint64_t salt){
 uint64_t key=rt_mix(salt^rt_mix(child));if(!e.p.mixed)return key;
 uint64_t group=signature(e.g.table[child],direction,extension_bucket(e,child)+17);
 return (rt_mix(salt^group)&0xffffffff00000000ULL)|(key&0xffffffffULL);
}
EP_HD inline int row_slot(int r,int64_t* hash,int size){
 int i=rt_mix(r)&(size-1);while(hash[i]!=-1&&(hash[i]>>32)!=r)i=(i+1)&(size-1);return i;
}
template<class Sink> RTC_RUN int visible_degree(Sink& e,int r,int direction){
 int degree=0;auto ptr=direction?e.pp:e.fp,dst=direction?e.pd:e.fd;
 int begin=ptr[r],full=ptr[r+1]-begin;
 int count=e.s.extension?rw_window_count(e.p,direction,
     e.fp[r+1]-e.fp[r],e.pp[r+1]-e.pp[r]):full;
 if(e.s.extension){e.s.rw_scanned+=count;e.s.rw_truncated+=full-count;}
 for(int k=0;k<count;++k){int j=e.s.extension?rw_scan_index(begin,full,count,k):begin+k;
  if((j==ptr[r]||dst[j]!=dst[j-1])&&row_ok(e.g,e.p,e.q,dst[j]))++degree;
 }
 return degree;
}
template<class Sink> RTC_RUN int64_t provenance(Sink& e,int r,int witness,bool filler){
 uint64_t sig=0;int hop=0;
 if(r!=e.q[0]){
  if(witness>=0){auto v=e.a.plan+PW*witness;sig=e.a.scratch[SW*v[2]];hop=v[3];}
  else {sig=signature(e.g.table[e.q[0]],filler?3:0,e.g.table[r]);hop=filler?65:1;
   if(e.p.own_recent&&!filler){int entity=e.p.forecast?int(e.m.entity[e.q[0]]):int(e.q[0]);
    if(entity>=0&&r!=entity)for(int j=e.pp[entity];j<e.pp[entity+1];++j)if(e.pd[j]==r){
     sig=e.p.forecast?signature(signature(e.g.table[e.q[0]],0,e.g.table[entity]),1,e.g.table[r]):signature(e.g.table[e.q[0]],1,e.g.table[r]);
     hop=e.p.forecast?2:1;break;}
   }
  }
  sig=1+sig%0x7fffffffULL;
 }
 int degree[2]={visible_degree(e,r,0),visible_degree(e,r,1)};
 uint64_t dt=uint8_t(delta_bucket(e.g.rowtime[r],e.q[1],r==e.q[0]));
 return sig|(uint64_t(hop)<<31)|(dt<<38)|(uint64_t(minimum(31,log_bucket(degree[1])))<<46)
        |(uint64_t(minimum(31,log_bucket(degree[0])))<<52);
}

template<class Sink> RTC_RUN bool leader(Sink&){return true;}
template<class Sink> RTC_RUN void sync(Sink&){}
template<class Sink> RTC_RUN int lane(Sink&){return 0;}
template<class Sink> RTC_RUN int lanes(Sink&){return 1;}
template<class Sink> RTC_RUN int stream_record(Sink& e,int parent,int direction,int64_t table){
 int canonical=parent<0?-1:e.a.scratch[SW*parent+18],lo=0,hi=e.s.schema_size;
 auto records=e.a.schema+e.s.schema_offset;
 while(lo<hi){int mid=lo+(hi-lo)/2;auto v=records+SCHEMA_WIDTH*mid;
  bool less=v[0]<canonical||(v[0]==canonical&&(v[1]<direction||(v[1]==direction&&v[2]<table)));
  if(less)lo=mid+1;else hi=mid;
 }
 if(lo==e.s.schema_size)return -1;
 auto v=records+SCHEMA_WIDTH*lo;
 if(v[0]!=canonical||v[1]!=direction||v[2]!=table)return -1;
 return lo;
}
template<class Sink> RTC_RUN int stream_prepared(Sink& e,int parent,int direction,int64_t table,int record){
 if(record<0)return -1;
 auto v=e.a.schema+e.s.schema_offset+SCHEMA_WIDTH*record;
 int id=v[3];if(e.a.mapping[id]>=0)return e.a.mapping[id];
 if(e.s.ns==e.p.S){e.s.error=8;return -1;}
 int sid=e.s.ns++;auto t=e.a.scratch+SW*sid;
 for(int j=0;j<SW;++j)t[j]=0;
 t[0]=v[4];t[1]=parent;t[2]=direction;t[3]=table;t[12]=parent<0?sid:e.a.scratch[SW*parent+12];
 t[18]=id;e.a.mapping[id]=sid;
 t[13]=-1;t[14]=-1;return sid;
}
template<class Sink> RTC_RUN int stream(Sink& e,int parent,int direction,int64_t table){
 return stream_prepared(e,parent,direction,table,stream_record(e,parent,direction,table));
}
template<class Sink> RTC_RUN bool payload_cell(Sink& e,int r,int64_t c,bool filler=false){
 return (r==e.q[0]&&c==e.m.target[r])||feature_ok(e.g,e.m,e.q,e.p,r,c)||
  (!filler&&c==e.m.target[r]&&label_ok(e.g,e.m,e.q,e.p,r));
}
template<class Sink> RTC_RUN int candidate_owned(Sink& e,int r,int direction,int64_t table){
 int owned=annotation(e.g,e.m,e.q,r)==1;
 if(e.m.kp[e.q[0]+1]-e.m.kp[e.q[0]]>1){
  owned=0;
  if(direction==0&&r!=e.q[0])
   for(auto i=e.m.kp[e.q[0]];i<e.m.kp[e.q[0]+1];++i)
    if(e.g.table[e.m.keys[i]]==table)
     for(auto j=e.m.kp[r];j<e.m.kp[r+1];++j)owned|=e.m.keys[i]==e.m.keys[j];
 }
 return owned;
}

template<class Sink> RTC_RUN bool pool_ok(Sink& e,int r){
 return label_ok(e.g,e.m,e.q,e.p,r)&&(!e.p.reserve||e.g.avail[e.m.target[r]]<e.q[1]);
}
template<class Sink> RTC_RUN bool candidate_payload(Sink& e,int r){
 for(auto c=e.g.cp[r];c<e.g.cp[r+1];++c)if(payload_cell(e,r,c))return true;
 return false;
}
template<class Sink> RTC_RUN void add_prepared(Sink& e,int r,int parent,int sid,int depth,int owned,bool payload,bool legal){
 auto& s=e.s;auto a=e.a;auto p=e.p;int64_t key=int64_t(sid)*p.N+r;int h=slot(key,a.seen,p.H);
 bool first=a.seen[h]==-1;
 if(!first&&depth>=p.h)return;
 if(first){if(s.hash_entries>=s.limit-1){a.scratch[SW*sid+8]=1;s.frontier_capped=1;return;}a.seen[h]=key;++s.hash_entries;}
 auto t=a.scratch+SW*sid;
 if(!legal){if(first)++t[7];return;}

 if(s.np>=s.limit){t[8]=1;s.frontier_capped=1;return;}
 int index=s.np++;auto v=a.plan+PW*index;
 v[0]=r;v[1]=parent;v[2]=sid;v[3]=depth;v[4]=-1;v[5]=0;
 v[6]=owned;t[19]|=owned;
 if(first&&payload){++t[4];++t[5];if(t[14]<0)t[13]=index;else a.plan[PW*t[14]+4]=index;t[14]=index;}
}
template<class Sink> RTC_RUN void add(Sink& e,int r,int parent,int sid,int depth){
 auto t=e.a.scratch+SW*sid,branch=e.a.scratch+SW*t[12];
 add_prepared(e,r,parent,sid,depth,candidate_owned(e,r,branch[2],branch[3]),
              candidate_payload(e,r),row_ok(e.g,e.p,e.q,r));
}

template<class Sink> RTC_RUN int admit(Sink& e,const int* rows,int n,bool filler=false,const int* witnesses=nullptr,int hidden=-1){
 auto& s=e.s;auto a=e.a;auto p=e.p;auto g=e.g;auto m=e.m;auto q=e.q;
 int needed=0,kenc=s.kenc;
 for(int i=0;i<n;++i){int r=rows[i];for(auto c=g.cp[r];c<g.cp[r+1];++c){
  if(!payload_cell(e,r,c,filler)||a.emitted[slot(c,a.emitted,p.E)]!=-1)continue;
  bool label=r!=q[0]&&c==m.target[r];if(label&&(kenc>=p.kmax||r==hidden))continue;kenc+=label;++needed;
 }}
 if(s.count+needed>e.settings[0])return -1;
 int end=s.count+needed;
 for(int i=0;i<n;++i){int r=rows[i],role=filler?4:annotation(g,m,q,r);
  for(auto c=g.cp[r];c<g.cp[r+1];++c){int h=slot(c,a.emitted,p.E);
   if(a.emitted[h]!=-1||!payload_cell(e,r,c,filler))continue;
   bool label=r!=q[0]&&c==m.target[r];if(label&&(s.kenc>=p.kmax||r==hidden))continue;
   int rh=row_slot(r,a.row_seen,p.E),start=boundary_offset(p)+MAX_SEQ_LEN;
   if(a.row_seen[rh]==-1){
    a.row_seen[rh]=(int64_t(r)<<32)|s.rows;
    if(p.provenance)a.receipt[start+s.rows]=provenance(e,r,witnesses?witnesses[i]:-1,filler);
    ++s.rows;
   }
   if(p.provenance)a.receipt[start+MAX_SEQ_LEN+s.count]=a.row_seen[rh]&0xffffffffULL;
   a.emitted[h]=c;a.cells[s.count]=c;a.roles[s.count]=role;a.receipt[boundary_offset(p)+s.count]=end;++s.count;
   if(s.extension)++a.table_counts[extension_group(e,r)];
   s.kenc+=label;s.targets+=r==q[0]&&c==m.target[r];s.text+=m.types[c]==1;s.filler+=role==4;s.own_cells+=role==1;
   s.peers+=label&&role==2;s.history+=label&&role==1;
  }

 }

 if(hidden>=0){int64_t c=m.target[hidden];if(c>=0){int h=slot(c,a.emitted,p.E);if(a.emitted[h]==-1)a.emitted[h]=c;}}
 return needed;
}

template<class Sink> RTC_RUN bool recent_before(Sink& e,int row,int other){
 if(row<0)return false;if(other<0)return true;
 int64_t t=e.g.rowtime[row],o=e.g.rowtime[other];
 return t>o||(t==o&&row<other);
}
template<class Sink> RTC_RUN int best_candidate(Sink& e,int sid){
 auto t=e.a.scratch+SW*sid;int best=-1;uint64_t bestkey=~uint64_t(0);
 for(int i=t[13];i>=0;i=e.a.plan[PW*i+4])if(!e.a.plan[PW*i+5]){
  int row=e.a.plan[PW*i];auto key=rt_mix(uint64_t(e.q[3])^row);
  if(e.s.extension?extension_before(e,row,best<0?-1:e.a.plan[PW*best]):
     (e.s.recent||e.p.canonical)?recent_before(e,row,best<0?-1:int(e.a.plan[PW*best])):
     (best<0||key<bestkey||(key==bestkey&&row<e.a.plan[PW*best]))){best=i;bestkey=key;}
 }
 return best;
}
template<class Sink> RTC_RUN int rw_neighbor(Sink& e,const int* path,int n,uint64_t salt){
 int row=path[n-1],best=-1;uint64_t best_key=~uint64_t(0);e.s.rw_last_truncated=0;
 for(int direction=0;direction<2;++direction){auto ptr=direction?e.pp:e.fp,dst=direction?e.pd:e.fd;
  int begin=ptr[row],degree=ptr[row+1]-begin;
  int count=rw_window_count(e.p,direction,
                            e.fp[row+1]-e.fp[row],e.pp[row+1]-e.pp[row]);
  if(degree>count){e.s.rw_truncated+=degree-count;e.s.rw_last_truncated=1;}
  e.s.rw_scanned+=count;
  for(int k=0;k<count;++k){int j=rw_scan_index(begin,degree,count,k),child=dst[j];bool cycle=false;
   for(int i=0;i<n;++i)cycle|=path[i]==child;
   if(cycle||!row_ok(e.g,e.p,e.q,child))continue;
   uint64_t key=rw_draw_key(e,direction,child,salt);
   if(best<0||key<best_key||(key==best_key&&child<best)){best=child;best_key=key;}
  }
 }
 return best;
}
template<class Sink> RTC_RUN void generate_rw(Sink& e,int64_t* visits){
 auto p=e.p;auto q=e.q;
 for(int w=0;w<p.W;++w){int path[65],n=1;path[0]=q[0];
  for(int step=0;step<p.K;++step){
   uint64_t salt=uint64_t(q[3])^rt_mix(w+1)^rt_mix(step+0x10001);
   int best=rw_neighbor(e,path,n,salt);if(best<0){if(!e.s.rw_last_truncated)++e.s.rw_graph_exhausted;break;}path[n++]=best;
   if(leader(e))visits[w*p.K+step]=best;
  }
 }
 sync(e);
}
struct RWSerial {
 RTGraph g;Meta m;Policy p;Work a;State& s;const int64_t *q,*settings;RTIndex32 fp,fd,pp,pd;
};
RTC_RUN inline void extend_rw(RWSerial& e,int64_t* visits){
 auto p=e.p;auto a=e.a;auto q=e.q;auto& s=e.s;
 s.blocked=0;
 int ranked=0;
 for(int i=0;i<p.W*p.K;++i){int row=visits[i];if(row<0)continue;bool seen=false;
  for(int j=0;j<i;++j)seen|=visits[j]==row;if(seen)continue;
  int count=0;for(int j=i;j<p.W*p.K;++j)count+=visits[j]==row;
  int at=0;
  if(p.mixed){uint64_t key=rt_mix(uint64_t(q[3])^row);
   while(at<ranked){int other=a.candidates[at];uint64_t old_key=rt_mix(uint64_t(q[3])^other);
    if(key<old_key||(key==old_key&&row<other))break;++at;
   }
  }else while(at<ranked&&(a.scores[at]>count||(a.scores[at]==count&&a.candidates[at]<row)))++at;
  if(at>=p.C)continue;ranked=minimum(p.C,ranked+1);
  for(int j=ranked-1;j>at;--j){a.candidates[j]=a.candidates[j-1];a.scores[j]=a.scores[j-1];}
  a.candidates[at]=row;a.scores[at]=count;
 }
 for(int step=0;step<ranked;++step){int candidate=step;
  if(p.mixed){candidate=-1;
   for(int i=0;i<ranked;++i)if(a.scores[i]>0&&extension_before(e,a.candidates[i],candidate<0?-1:int(a.candidates[candidate])))candidate=i;
   if(candidate<0)break;a.scores[candidate]=-a.scores[candidate];
  }
  int best=a.candidates[candidate],witness=-1;
  for(int i=0;i<p.W*p.K;++i)if(visits[i]==best&&
      (witness<0||i%p.K<witness%p.K))witness=i;
  int rows[65],n=1;rows[0]=q[0];
  int start=(witness/p.K)*p.K;
  for(int i=start;i<=witness;++i)rows[n++]=visits[i];
  int old_rows=s.rows;
  if(admit(e,rows,n)<0){s.blocked=1;continue;}
  if(p.provenance){uint64_t sig=e.g.table[q[0]];
   for(int i=1;i<n;++i){int previous=rows[i-1],r=rows[i],direction=1;
    for(int j=e.fp[previous];j<e.fp[previous+1];++j)if(e.fd[j]==r){direction=0;break;}
    sig=signature(sig,direction,e.g.table[r]);int h=row_slot(r,a.row_seen,p.E);
    if(a.row_seen[h]<0)continue;int slot_id=a.row_seen[h]&0xffffffffULL;
    if(slot_id>=old_rows){auto& word=a.receipt[boundary_offset(p)+MAX_SEQ_LEN+slot_id];
     word=(uint64_t(word)&~((uint64_t(1)<<38)-1))|(1+sig%0x7fffffffULL)|(uint64_t(i)<<31);
    }
   }
  }
 }
 if(p.mixed)for(int i=0;i<ranked;++i)if(a.scores[i]<0)a.scores[i]=-a.scores[i];
}

constexpr int SEED_POOL_FACTOR=4, SEED_LOCAL_SCAN=64, SEED_CLASS_CAP=4095, SEED_LOCAL_TABLES=7;
constexpr int ROUTE_CHILDREN=256, ROUTE_TASKS=4, ROUTE_SCAN=64;
constexpr int64_t RTC_STATIC=-(int64_t(1)<<62);
template<class Sink> RTC_RUN bool seed_ok(Sink& e,int r){
 if(!label_ok(e.g,e.m,e.q,e.p,r))return false;
 int64_t c=e.m.target[r];
 return e.g.avail[c]<e.q[1]&&e.a.emitted[slot(c,e.a.emitted,e.p.E)]==-1;
}
template<class Sink> RTC_RUN bool seed_before(Sink& e,int row,int64_t count,int other,int64_t other_count){
 if(count!=other_count)return count>other_count;
 int64_t t=e.g.rowtime[row],o=e.g.rowtime[other];
 if(t!=o)return t>o;
 return row<other;
}
template<class Sink> RTC_RUN void seed_rank(Sink& e,int row,int64_t count,int& ranked){
 auto a=e.a;int at=0;
 while(at<ranked&&!seed_before(e,row,count,int(a.candidates[at]),a.scores[at]))++at;
 if(at>=e.p.C)return;
 ranked=minimum(e.p.C,ranked+1);
 for(int j=ranked-1;j>at;--j){a.candidates[j]=a.candidates[j-1];a.scores[j]=a.scores[j-1];}
 a.candidates[at]=row;a.scores[at]=count;
}

template<class Sink> RTC_RUN int children_until(Sink& e,int row,int64_t limit){
 int lo=e.pp[row],hi=e.pp[row+1];if(limit==RTC_STATIC)return hi;
 while(lo<hi){int mid=lo+(hi-lo)/2;if(e.g.rowtime[e.pd[mid]]<=limit)lo=mid+1;else hi=mid;}
 return lo;
}
template<class Sink> RTC_RUN bool seed_local_ok(Sink& e,int x,int64_t ts,const int* rows,int n){
 if(x==e.q[0]||e.g.task[x]!=0||!row_ok(e.g,e.p,e.q,x))return false;
 int64_t t=e.g.rowtime[x];
 if(ts!=RTC_STATIC&&t!=RTC_STATIC&&t>ts)return false;
 for(int i=0;i<n;++i)if(rows[i]==x)return false;
 return true;
}
template<class Sink> RTC_RUN int table_population(Sink& e,int64_t table){
 int lo=0,hi=e.p.T;
 while(lo<hi){int mid=(lo+hi)/2;if(e.m.table_ids[mid]<table)lo=mid+1;else hi=mid;}
 return lo<e.p.T?int(e.m.caps[lo]):0;
}
template<class Sink> RTC_RUN int seed_bundle(Sink& e,int seed,int* rows,int limit,int policy){
 int n=1;rows[0]=seed;if(!limit)return n;
 int64_t ts=e.g.rowtime[seed];bool largest=policy==2;
 if(policy){
  int entity=e.p.forecast?int(e.m.entity[seed]):seed;if(entity<0)return n;
  if(entity!=seed&&seed_local_ok(e,entity,ts,rows,n))rows[n++]=entity;
  int begin=e.pp[entity],end=e.pp[entity+1],hi=children_until(e,entity,ts);
  int64_t tables[SEED_LOCAL_TABLES];int pops[SEED_LOCAL_TABLES],nt=0;
  for(int j=hi-1,scanned=0;j>=begin&&scanned<SEED_LOCAL_SCAN;--j,++scanned){int y=e.pd[j];
   if(e.g.task[y]||!row_ok(e.g,e.p,e.q,y))continue;int64_t table=e.g.table[y];bool known=false;
   for(int k=0;k<nt;++k)known|=tables[k]==table;if(known||nt==SEED_LOCAL_TABLES)continue;
   int pop=table_population(e,table),at=nt;
   while(at>0&&((largest?pops[at-1]<pop:pops[at-1]>pop)||(pops[at-1]==pop&&tables[at-1]>table))){tables[at]=tables[at-1];pops[at]=pops[at-1];--at;}
   tables[at]=table;pops[at]=pop;++nt;
  }
  for(int round=0;round<limit&&n<=limit;++round)
   for(int k=0;k<nt&&n<=limit;++k){int seen=0;
    for(int j=hi-1,scanned=0;j>=begin&&scanned<SEED_LOCAL_SCAN;--j,++scanned){int y=e.pd[j];
     if(j+1<end&&e.pd[j+1]==y)continue;
     if(e.g.table[y]!=tables[k]||e.g.task[y]||!row_ok(e.g,e.p,e.q,y))continue;
     if(seen++<round)continue;
     if(seed_local_ok(e,y,ts,rows,n))rows[n++]=y;
     break;
    }
   }
  return n;
 }
 for(int j=e.fp[seed];j<e.fp[seed+1]&&n<=limit;++j){int x=e.fd[j];
  if(j>e.fp[seed]&&e.fd[j-1]==x)continue;
  if(seed_local_ok(e,x,ts,rows,n))rows[n++]=x;
 }
 int parents=n;
 for(int i=1;i<parents&&n<=limit;++i){int x=rows[i],begin=e.pp[x],end=e.pp[x+1],hi=children_until(e,x,ts);
  for(int j=hi-1,scanned=0;j>=begin&&scanned<SEED_LOCAL_SCAN&&n<=limit;--j,++scanned){int y=e.pd[j];
   if(j+1<end&&e.pd[j+1]==y)continue;
   if(seed_local_ok(e,y,ts,rows,n))rows[n++]=y;
  }
 }
 return n;
}

template<class Sink> RTC_RUN void route_count(Sink& e,int64_t* visits,int& distinct,int row){
 int n=e.p.W*e.p.K;int i=int(rt_mix(uint64_t(row))%uint64_t(n));
 while(visits[i]>=0&&int(visits[i]>>16)!=row)i=(i+1)%n;
 if(visits[i]<0){if(4*(distinct+1)>3*n)return;visits[i]=int64_t(row)<<16;++distinct;}
 if((visits[i]&0xffff)<0xffff)++visits[i];
}

template<class Sink> RTC_RUN void route_seeds(Sink& e,int other,int64_t* visits,int& distinct){
 int64_t root_table=e.g.table[e.q[0]];
 if(!e.p.forecast){if(e.g.table[other]==root_table&&seed_ok(e,other))route_count(e,visits,distinct,other);return;}
 int begin=e.pp[other],end=e.pp[other+1],hi=children_until(e,other,e.q[1]),found=0;
 for(int j=hi-1,scanned=0;j>=begin&&scanned<ROUTE_SCAN&&found<ROUTE_TASKS;--j,++scanned){int t=e.pd[j];
  if(j+1<end&&e.pd[j+1]==t)continue;
  if(e.g.table[t]!=root_table||!seed_ok(e,t))continue;
  route_count(e,visits,distinct,t);++found;
 }
}

EP_HD inline int route_position(int begin,int count,int size,int k){
 return begin+int((uint64_t(2)*uint64_t(k)+1)*uint64_t(count)/(uint64_t(2)*uint64_t(size)));
}

template<class Sink> RTC_RUN void shared_routes(Sink& e,int64_t* visits){
 int distinct=0,root=e.q[0],fanout=e.p.seed_fanout;
 int entity=e.p.forecast?int(e.m.entity[root]):root;if(entity<0)return;
 int64_t etable=e.g.table[entity],cut=e.q[1];
 for(int j=e.fp[entity];j<e.fp[entity+1];++j){int parent=e.fd[j];
  if(j>e.fp[entity]&&e.fd[j-1]==parent)continue;
  if(e.g.task[parent]||!row_ok(e.g,e.p,e.q,parent))continue;
  int begin=e.pp[parent],count=children_until(e,parent,cut)-begin,size=minimum(count,fanout);
  for(int k=0;k<size;++k){int other=e.pd[route_position(begin,count,size,k)];
   if(other==entity||e.g.table[other]!=etable||!row_ok(e.g,e.p,e.q,other))continue;
   route_seeds(e,other,visits,distinct);
  }
 }
 int begin=e.pp[entity],end=e.pp[entity+1];
 for(int k=children_until(e,entity,cut)-1,scanned=0;k>=begin&&scanned<ROUTE_CHILDREN;--k,++scanned){int event=e.pd[k];
  if(k+1<end&&e.pd[k+1]==event)continue;
  if(e.g.task[event]||!row_ok(e.g,e.p,e.q,event))continue;
  for(int j=e.fp[event];j<e.fp[event+1];++j){int member=e.fd[j];
   if(j>e.fp[event]&&e.fd[j-1]==member)continue;
   if(member==entity||e.g.table[member]==etable||e.g.task[member]||!row_ok(e.g,e.p,e.q,member))continue;
   int mb=e.pp[member],count=children_until(e,member,cut)-mb,size=minimum(count,fanout);
   for(int x=0;x<size;++x){int peer=e.pd[route_position(mb,count,size,x)];
    if(peer==event||e.g.table[peer]!=e.g.table[event]||e.g.task[peer]||!row_ok(e.g,e.p,e.q,peer))continue;
    for(int y=e.fp[peer];y<e.fp[peer+1];++y){int other=e.fd[y];
     if(y>e.fp[peer]&&e.fd[y-1]==other)continue;
     if(other==entity||e.g.table[other]!=etable||!row_ok(e.g,e.p,e.q,other))continue;
     route_seeds(e,other,visits,distinct);
    }
   }
  }
 }
}

EP_HD inline int64_t torch_mix(int64_t x){

 x=int64_t(uint64_t(x)+0x9e3779b97f4a7c15ULL);
 x=int64_t(uint64_t(x^(x>>30))*0xbf58476d1ce4e5b9ULL);
 x=int64_t(uint64_t(x^(x>>27))*0x94d049bb133111ebULL);
 return x^(x>>31);
}

EP_HD inline bool triple_less(const int64_t* a,const int64_t* b,bool by_index){
 if(by_index)return a[1]<b[1];
 return a[0]<b[0]||(a[0]==b[0]&&a[1]<b[1]);
}
EP_HD inline void triple_swap(int64_t* a,int64_t* b){for(int i=0;i<3;++i){int64_t t=a[i];a[i]=b[i];b[i]=t;}}
EP_HD inline void triple_sift(int64_t* h,int n,int i,bool by_index){
 while(2*i+1<n){int c=2*i+1;if(c+1<n&&triple_less(h+3*c,h+3*(c+1),by_index))++c;
  if(!triple_less(h+3*i,h+3*c,by_index))break;triple_swap(h+3*i,h+3*c);i=c;}
}
template<class Sink> RTC_RUN int program_hop(Sink& e,const int64_t* in,int nin,int direction,int64_t table,
                                             bool allow_task,int64_t* out,int64_t* heap){
 int cap=e.p.seed_hop_cap,width=minimum(maximum(2*cap,64),4096),n=0;int64_t index=0,cut=e.q[1];
 for(int f=0;f<nin;++f){int r=int(in[f]);auto ptr=direction?e.pp:e.fp;auto dst=direction?e.pd:e.fd;
  int begin=ptr[r],end=ptr[r+1];

  if(e.p.seed_program_mature&&direction)end=children_until(e,r,cut-1);
  int scan=minimum(end-begin,width);
  for(int off=0;off<scan;++off){int x=dst[end-1-off];
   if(e.g.table[x]!=table||(!allow_task&&e.g.task[x]!=0))continue;
   int64_t t=e.g.rowtime[x];if(t!=RTC_STATIC&&t>=cut)continue;
   if(e.p.seed_program_mature&&allow_task){int64_t c=e.m.target[x];if(c<0||e.g.avail[c]>=cut)continue;}
   int64_t item[3]={torch_mix(e.q[3]^int64_t(x)),index++,x};
   if(n<cap){int i=n++;for(int j=0;j<3;++j)heap[3*i+j]=item[j];
    while(i>0){int up=(i-1)/2;if(!triple_less(heap+3*up,heap+3*i,false))break;triple_swap(heap+3*up,heap+3*i);i=up;}
   }else if(triple_less(item,heap,false)){for(int j=0;j<3;++j)heap[j]=item[j];triple_sift(heap,n,0,false);}
  }
 }

 for(int i=n/2-1;i>=0;--i)triple_sift(heap,n,i,true);
 for(int i=n-1;i>0;--i){triple_swap(heap,heap+3*i);triple_sift(heap,i,0,true);}
 for(int i=0;i<n;++i)out[i]=heap[3*i+2];
 return n;
}
template<class Sink> RTC_RUN void program_routes(Sink& e,int& ranked){
 auto a=e.a;int cap=e.p.seed_hop_cap,root=e.q[0];
 int64_t *heap=a.seed_work,*front=heap+3*cap,*back=front+cap,*finals=back+cap;int nfinals=0,paths=0;
 int64_t etable=e.p.forecast?e.g.table[e.m.entity[root]]:e.g.table[root],ttable=e.g.table[root];
 for(int step=0;step<e.s.ns&&paths<PROGRAM_PATHS;++step){int sid=int(a.mapping[step]);int cls=quota_class(e,sid);
  if(cls!=0&&cls!=2)continue;auto t=a.scratch+SW*sid;++paths;
  int dirs[6],hops=0;int64_t tables[6];
  if(e.p.forecast){dirs[hops]=0;tables[hops++]=etable;}
  if(cls==2){int64_t c=a.scratch[SW*t[1]+3];
   dirs[hops]=1;tables[hops++]=c;dirs[hops]=0;tables[hops++]=t[3];dirs[hops]=1;tables[hops++]=c;dirs[hops]=0;tables[hops++]=etable;}
  else{dirs[hops]=0;tables[hops++]=t[3];dirs[hops]=1;tables[hops++]=etable;}
  if(e.p.forecast){dirs[hops]=1;tables[hops++]=ttable;}
  front[0]=root;int n=1;
  for(int h=0;h<hops&&n;++h){n=program_hop(e,front,n,dirs[h],tables[h],h==hops-1,back,heap);
   int64_t* swap=front;front=back;back=swap;}
  for(int i=0;i<n;++i)finals[nfinals++]=front[i];
  front=heap+3*cap;back=front+cap;
 }

 for(int i=nfinals/2-1;i>=0;--i){int at=i;while(2*at+1<nfinals){int c=2*at+1;if(c+1<nfinals&&finals[c]<finals[c+1])++c;
  if(finals[at]>=finals[c])break;int64_t x=finals[at];finals[at]=finals[c];finals[c]=x;at=c;}}
 for(int i=nfinals-1;i>0;--i){int64_t x=finals[0];finals[0]=finals[i];finals[i]=x;int at=0;
  while(2*at+1<i){int c=2*at+1;if(c+1<i&&finals[c]<finals[c+1])++c;if(finals[at]>=finals[c])break;
   int64_t y=finals[at];finals[at]=finals[c];finals[c]=y;at=c;}}
 for(int i=0;i<nfinals;){int j=i;while(j<nfinals&&finals[j]==finals[i])++j;
  int row=int(finals[i]);if(seed_ok(e,row))seed_rank(e,row,j-i,ranked);i=j;}
}
template<class Sink> RTC_RUN void seed_tier(Sink& e,int64_t* visits){
 auto p=e.p;auto a=e.a;auto& s=e.s;
 int ranked=0,n=p.W*p.K;
 if(p.seed_rank==2)program_routes(e,ranked);
 else if(p.seed_rank){
  shared_routes(e,visits);s.walked=2;
  for(int i=0;i<n;++i){if(visits[i]<0)continue;int row=int(visits[i]>>16);
   if(seed_ok(e,row))seed_rank(e,row,visits[i]&0xffff,ranked);}
 }else{
  generate_rw(e,visits);s.walked=1;
  for(int i=0;i<n;++i){int row=visits[i];if(row<0)continue;bool seen=false;
   for(int j=0;j<i;++j)seen|=visits[j]==row;if(seen||!seed_ok(e,row))continue;
   int count=0;for(int j=i;j<n;++j)count+=visits[j]==row;
   seed_rank(e,row,count,ranked);
  }
 }
 if(p.seed_fill&&s.pool_sid){int sid=s.pool_sid-1,considered=0;
  for(int i=1;i<s.np&&considered<SEED_POOL_FACTOR*p.C;++i){auto v=a.plan+PW*i;if(v[2]!=sid)continue;++considered;
   int row=v[0];if(!seed_ok(e,row))continue;bool dup=false;
   for(int j=0;j<ranked;++j)dup|=a.candidates[j]==row;
   if(!dup)seed_rank(e,row,0,ranked);
  }
 }
 if(p.seed_balance&&ranked>1){

  for(int i=0;i<ranked;++i){int64_t cls=minimum(SEED_CLASS_CAP,int(e.m.classes[a.candidates[i]]));int64_t k=0;
   for(int j=0;j<i;++j)k+=minimum(SEED_CLASS_CAP,int(e.m.classes[a.candidates[j]]))==cls;
   a.scores[i]|=(k<<32)|(cls<<16);
  }
  for(int i=1;i<ranked;++i){int64_t row=a.candidates[i],score=a.scores[i];int j=i;
   while(j>0&&(a.scores[j-1]>>16)>(score>>16)){a.candidates[j]=a.candidates[j-1];a.scores[j]=a.scores[j-1];--j;}
   a.candidates[j]=row;a.scores[j]=score;
  }
  for(int i=0;i<ranked;++i)a.scores[i]&=0xffff;
 }
 for(int k=0;k<ranked&&s.count<e.settings[0];++k){int rows[65];int seed=int(a.candidates[k]);
  int m=seed_bundle(e,seed,rows,p.seed_local,p.seed_local_policy);admit(e,rows,m,false,nullptr,p.seed_hide?seed:-1);}
}

template<class Sink> RTC_RUN void peer_extension(Sink& e){
 auto a=e.a;auto p=e.p;
 for(int k=0;k<p.C&&a.candidates[k]>=0&&e.s.count<e.settings[0];++k){int rows[65];int seed=int(a.candidates[k]);
  int m=seed_bundle(e,seed,rows,p.peer_ext_rows,2);admit(e,rows,m,false,nullptr,p.seed_hide?seed:-1);}
}

template<class Sink> RTC_RUN void own_label_tier(Sink& e,int scale){
 if(!e.p.forecast)return;
 int entity=e.m.entity[e.q[0]];if(entity<0)return;
 int budget=e.p.own_labels*scale,taken=0;
 for(int64_t i=e.m.hp[entity+1]-1;i>=e.m.hp[entity]&&taken<budget;--i){int r=e.m.hr[i];
  if(!label_ok(e.g,e.m,e.q,e.p,r))continue;
  int rows[1]={r};if(admit(e,rows,1)<0)break;++taken;
 }
}

template<class Sink> RTC_RUN bool own_recent_candidate(Sink& e,int r){
 if(r==e.q[0]||e.a.row_seen[row_slot(r,e.a.row_seen,e.p.E)]!=-1||!row_ok(e.g,e.p,e.q,r))return false;
 if(e.g.task[r])return label_ok(e.g,e.m,e.q,e.p,r);
 return candidate_payload(e,r);
}

template<class Sink> RTC_RUN bool own_source_before(Sink& e,int row,int other){
 if(!e.m.source_rank)return recent_before(e,row,other);
 if(row<0)return false;if(other<0)return true;
 int64_t t=e.m.source_time[row],o=e.m.source_time[other];
 return t>o||(t==o&&row<other);
}

template<class Sink> RTC_RUN void own_recent_rounds(Sink& e,int entity){
 int last[64];
 for(int ti=0;ti<e.p.T;++ti)last[ti]=-1;
 for(int taken=0;taken<e.settings[0];){
  bool any=false;
  for(int ti=0;ti<e.p.T&&taken<e.settings[0];++ti){
   if(last[ti]==-2)continue;
   int64_t table=e.m.table_ids[ti];int pick=-1;
   int key_count=e.m.source_rank?int(e.m.fp[e.q[0]+1]-e.m.fp[e.q[0]])+1:1;
   for(int ki=0;ki<key_count;++ki){
    int key=e.m.source_rank?(ki==0?int(e.q[0]):int(e.m.fd[e.m.fp[e.q[0]]+ki-1])):entity;
    for(int j=e.pp[key];j<e.pp[key+1];++j){int r=e.pd[j];
    if(e.g.table[r]!=table||!own_recent_candidate(e,r))continue;
    if(last[ti]>=0&&!own_source_before(e,last[ti],r))continue;
    if(own_source_before(e,r,pick))pick=r;}}
   if(pick>=0)last[ti]=pick;
   if(pick<0)continue;
   any=true;int rows[1]={pick};
   if(admit(e,rows,1)<0){

    if(e.m.source_rank){last[ti]=-2;continue;}
    return;
   }
   ++taken;
  }
  if(!any)return;
 }
}
template<class Sink> RTC_RUN void own_recent_tier(Sink& e){
 int entity=e.p.forecast?int(e.m.entity[e.q[0]]):int(e.q[0]);
 if(e.p.own_recent==4){
  // Label-rich order: every admitted label row ranked by (own entity, time, row), then the usual own-recency fill.
  int ranked=0,pool_size=e.m.pool_size<0?e.p.N:e.m.pool_size;
  for(int i=0;i<pool_size;++i){int r=e.m.pool_size<0?i:int(e.m.pool[i]);
   if(!seed_ok(e,r))continue;
   seed_rank(e,r,entity>=0&&e.m.entity[r]==entity?1:0,ranked);
  }
  for(int i=0;i<ranked;++i){int rows[1]={int(e.a.candidates[i])};
   if(admit(e,rows,1)<0)break;
  }
 }
 if(e.p.own_recent==3){

  int last[64],served[64];
  for(int ti=0;ti<e.p.T;++ti){last[ti]=-1;served[ti]=0;}
  for(int taken=0;taken<e.settings[0];){
   int chosen=-1,pick=-1;
   for(int ti=0;ti<e.p.T;++ti){
    if(last[ti]==-2||(chosen>=0&&served[ti]>=served[chosen]))continue;
    int candidate=-1;
    for(int ki=0;ki<int(e.m.fp[e.q[0]+1]-e.m.fp[e.q[0]])+1;++ki){
     int key=ki==0?int(e.q[0]):int(e.m.fd[e.m.fp[e.q[0]]+ki-1]);
     for(int j=e.pp[key];j<e.pp[key+1];++j){int r=e.pd[j];
      if(e.g.table[r]!=e.m.table_ids[ti]||!own_recent_candidate(e,r))continue;
      if(last[ti]>=0&&!own_source_before(e,last[ti],r))continue;
      if(own_source_before(e,r,candidate))candidate=r;
     }
    }
    if(candidate<0){last[ti]=-2;continue;}
    chosen=ti;pick=candidate;
   }
   if(chosen<0)return;
   last[chosen]=pick;int rows[1]={pick},charged=admit(e,rows,1);
   if(charged<0){last[chosen]=-2;continue;}
   served[chosen]+=charged;++taken;
  }
  return;
 }
 if(e.m.source_rank&&e.p.own_recent==2){own_recent_rounds(e,entity);return;}
 if(entity<0)return;
 if(e.p.own_recent==2){own_recent_rounds(e,entity);return;}
 if(e.m.recent.data){
  int begin=e.pp[entity],end=e.pp[entity+1];

  for(int ti=0;ti<e.p.T;++ti){
   int64_t table=e.m.table_ids[ti];
   for(int j=begin;j<end;++j){int r=e.m.recent[j];
    if(e.g.table[r]!=table||!own_recent_candidate(e,r))continue;
    int rows[1]={r};if(admit(e,rows,1)<0)return;break;
   }
  }
  for(int j=begin,taken=0;j<end&&taken<e.settings[0];++j){int r=e.m.recent[j];
   if(!own_recent_candidate(e,r))continue;
   int rows[1]={r};int charged=admit(e,rows,1);

   if(charged<=0)return;
   ++taken;
  }
  return;
 }

 for(int ti=0;ti<e.p.T;++ti){
  int64_t table=e.m.table_ids[ti];
  int best=-1;
  for(int j=e.pp[entity];j<e.pp[entity+1];++j){int r=e.pd[j];
   if(e.g.table[r]==table&&own_recent_candidate(e,r)&&recent_before(e,r,best))best=r;}
  if(best>=0){int rows[1]={best};if(admit(e,rows,1)<0)return;}
 }
 for(int taken=0;taken<e.settings[0];++taken){int best=-1;
  for(int j=e.pp[entity];j<e.pp[entity+1];++j){int r=e.pd[j];
   if(own_recent_candidate(e,r)&&recent_before(e,r,best))best=r;
  }
  if(best<0)return;int rows[1]={best};if(admit(e,rows,1)<0)return;
 }
}
template<class Sink> RTC_RUN void consume_candidate(Sink&,int){}
template<class Sink> RTC_RUN bool take(Sink& e,int sid){
 auto t=e.a.scratch+SW*sid;int best=best_candidate(e,sid);
 if(best<0)return true;
 int reversed[66],rows[66],witnesses[66],n=0;
 for(int i=best;i>=0;i=e.a.plan[PW*i+1]){if(n==66){if(leader(e))e.s.error=8;sync(e);return false;}reversed[n++]=i;}
 for(int i=0;i<n;++i){witnesses[i]=reversed[n-i-1];rows[i]=e.a.plan[PW*witnesses[i]];}
 int charged=admit(e,rows,n,false,witnesses);
 if(charged<0){
  if(e.s.extension){if(leader(e)){e.a.plan[PW*best+5]=1;++t[9];consume_candidate(e,sid);}sync(e);}
  return false;
 }
 if(leader(e)){e.a.plan[PW*best+5]=1;++t[6];t[10]+=charged;consume_candidate(e,sid);}
 sync(e);return true;
}

template<class Sink> RTC_RUN void signature_sift(Sink& e,int root,int n){
 auto order=e.a.mapping;auto scratch=e.a.scratch;
 while(2*root+1<n){int child=2*root+1;
  if(child+1<n&&scratch[SW*order[child]]<scratch[SW*order[child+1]])++child;
  if(scratch[SW*order[root]]>=scratch[SW*order[child]])break;
  auto tmp=order[root];order[root]=order[child];order[child]=tmp;root=child;
 }
}
template<class Sink> RTC_RUN void signature_order(Sink& e){
 auto order=e.a.mapping;int n=e.s.ns;
 for(int i=0;i<n;++i)order[i]=i;
 for(int i=n/2;i>0;--i)signature_sift(e,i-1,n);
 for(int i=n-1;i>0;--i){auto tmp=order[0];order[0]=order[i];order[i]=tmp;signature_sift(e,0,i);}
}
template<class Sink> RTC_RUN void setup(Sink& e){
 auto& s=e.s;auto a=e.a;auto p=e.p;auto g=e.g;auto m=e.m;auto q=e.q;int root=q[0];
#ifndef __CUDA_ARCH__

 for(int i=0;i<receipt_size(p);++i)a.receipt[i]=i>=STREAM_OFFSET?-1:0;
 for(int i=0;i<p.H;++i)a.seen[i]=-1;
 for(int i=0;i<p.S;++i)a.mapping[i]=-1;
#endif
 if(m.task_capacity[0]<p.seed_rows||m.task_capacity[0]>p.F){s.error=8;return;}
 s.frontier=frontier_bound(p,int(m.task_capacity[0]),e.settings[4]);
 s.limit=s.frontier-p.reserve;
 int lo=0,hi=p.T;
 while(lo<hi){int mid=lo+(hi-lo)/2;if(a.schema[SCHEMA_HEADER*mid]<g.table[root])lo=mid+1;else hi=mid;}
 if(lo==p.T||a.schema[SCHEMA_HEADER*lo]!=g.table[root]){s.error=8;return;}

 if(!m.stream_mask[lo]){s.error=9;return;}
 auto header=a.schema+SCHEMA_HEADER*lo;
 s.schema_offset=header[1];s.schema_size=header[2];s.total=header[3];s.pruned=header[4];s.saturated=header[5];

 if(p.weight_policy)for(int i=0;i<s.schema_size;++i){int64_t scale=a.schema[s.schema_offset+SCHEMA_WIDTH*i+5];
  if(scale<1||scale>(int64_t(1)<<20)){s.error=10;return;}}
 if(m.target[root]<0||g.col[m.target[root]]!=q[2]){s.error=3;return;}
 if(p.forecast&&m.entity[root]<0){s.error=4;return;}
 if(p.forecast){int entity=m.entity[root];bool visible=false;
  for(auto c=g.cp[entity];c<g.cp[entity+1];++c)if(payload_cell(e,entity,c)){visible=true;break;}
  // A null-only entity has no archived cells but still defines a query.
  bool empty=g.cp[entity]==g.cp[entity+1]&&row_ok(g,p,q,entity);
  if(!visible&&!empty){s.error=4;return;}
 }
}
template<class Sink> RTC_RUN void seed_ego(Sink& e){
 auto& s=e.s;auto a=e.a;auto p=e.p;auto g=e.g;auto m=e.m;auto q=e.q;int root=q[0];

 a.plan[0]=root;a.plan[3]=0;s.ego_rows=1;a.seen[slot(root,a.seen,p.H)]=root;
 for(int cursor=0;cursor<s.ego_rows;++cursor){int r=a.plan[PW*cursor],depth=a.plan[PW*cursor+3];
  if(depth>=p.seed_hops)continue;
  for(int direction=0;direction<2;++direction){auto ptr=direction?e.pp:e.fp,dst=direction?e.pd:e.fd;
   for(int k=0;k<ptr[r+1]-ptr[r];++k){int j=p.canonical?ptr[r+1]-1-k:ptr[r]+k;int child=dst[j],h=slot(child,a.seen,p.H);
    if(j>ptr[r]&&dst[j-1]==child)continue;
    if(a.seen[h]!=-1||!row_ok(g,p,q,child))continue;
    if(s.ego_rows==p.seed_rows){++s.ego_cap;continue;}
    a.seen[h]=child;a.plan[PW*s.ego_rows]=child;a.plan[PW*s.ego_rows+3]=depth+1;++s.ego_rows;
   }
  }
 }

 for(int i=0;i<s.ego_rows;++i)a.plan[PW*i+1]=slot(a.plan[PW*i],a.seen,p.H);
 for(int i=0;i<s.ego_rows;++i)a.seen[a.plan[PW*i+1]]=-1;
}
template<class Sink> RTC_RUN void discover(Sink& e){
 auto& s=e.s;auto a=e.a;auto p=e.p;auto g=e.g;auto m=e.m;auto q=e.q;int root=q[0];
 a.plan[0]=root;a.plan[1]=-1;a.plan[2]=-1;a.plan[3]=0;a.plan[4]=-1;a.plan[5]=0;a.plan[6]=0;s.np=1;
 for(int cursor=0;cursor<s.np&&!s.error&&!s.frontier_capped;++cursor){auto v=a.plan+PW*cursor;int r=v[0],depth=v[3],parent_sid=v[2];
  if(depth>=p.h)continue;
  for(int direction=0;direction<2&&!s.error&&!s.frontier_capped;++direction){auto ptr=direction?e.pp:e.fp,dst=direction?e.pd:e.fd;
   for(int k=0;k<ptr[r+1]-ptr[r]&&!s.error&&!s.frontier_capped;++k){int j=p.canonical?ptr[r+1]-1-k:ptr[r]+k;int child=dst[j];bool cycle=false;
    if(j>ptr[r]&&dst[j-1]==child)continue;
    for(int i=cursor;i>=0;i=a.plan[PW*i+1])cycle|=a.plan[PW*i]==child;
    if(cycle||!structural(g,p,q,child)||(p.canonical&&!row_ok(g,p,q,child)))continue;
    int sid=stream(e,parent_sid,direction,g.table[child]);if(s.error)break;if(sid>=0)add(e,child,cursor,sid,depth+1);
   }
  }
 }
 if(s.error)return;

 if(p.reserve){s.limit=s.frontier;s.frontier_capped=0;}
 int pool=-1;
 int pool_size=m.pool_size<0?p.N:m.pool_size;
 for(int k=0;k<pool_size&&!s.error&&!s.frontier_capped;++k){int i=p.reserve?pool_size-1-k:k;int r=m.pool_size<0?i:int(m.pool[i]);if(!pool_ok(e,r))continue;
  if(pool<0){pool=stream(e,-1,2,g.table[root]);if(pool>=0)s.pool_sid=pool+1;}if(s.error)break;if(pool>=0)add(e,r,0,pool,p.h);
 }
 if(s.error)return;
}
template<class Sink> RTC_RUN void prepare_streams(Sink& e){
 auto& s=e.s;auto a=e.a;auto p=e.p;auto g=e.g;auto m=e.m;auto q=e.q;int root=q[0];

 if(!p.weight_policy){for(int i=0;i<s.ns;++i){auto t=a.scratch+SW*i;t[11]=isqrt(uint64_t(t[4])*(1<<20));}}
 else{auto records=a.schema+s.schema_offset;
  for(int rec=0;rec<s.schema_size;++rec){auto v=records+SCHEMA_WIDTH*rec;int sid=int(a.mapping[v[3]]);
   if(sid>=0){auto t=a.scratch+SW*sid;t[11]=isqrt(uint64_t(t[4])*uint64_t(v[5]));}}}

 for(int i=0;i<s.ns;++i){auto t=a.scratch+SW*i;
  a.scratch[SW*t[12]+16]+=p.weight_policy?isqrt(uint64_t(t[4])*(1<<20)):t[11];}
 signature_order(e);
}

template<class Sink> RTC_RUN bool entity_stream(Sink& e,int sid){
 if(!e.p.forecast)return sid<0;
 if(sid<0)return false;auto t=e.a.scratch+SW*sid;
 return t[1]<0&&t[2]==0&&t[3]==e.g.table[e.m.entity[e.q[0]]];
}
template<class Sink> RTC_RUN int64_t entity_table(Sink& e){
 return e.p.forecast?e.g.table[e.m.entity[e.q[0]]]:e.g.table[e.q[0]];
}
template<class Sink> RTC_RUN bool own_event_stream(Sink& e,int sid){
 auto t=e.a.scratch+SW*sid;
 return t[2]==1&&entity_stream(e,int(t[1]))&&t[3]!=e.g.table[e.q[0]];
}
template<class Sink> RTC_RUN bool parent_stream(Sink& e,int sid){
 auto t=e.a.scratch+SW*sid;
 return t[2]==0&&entity_stream(e,int(t[1]))&&t[3]!=entity_table(e);
}
template<class Sink> RTC_RUN int quota_class(Sink& e,int sid){
 auto t=e.a.scratch+SW*sid;if(t[2]==2)return -1;
 if(own_event_stream(e,sid))return 1;
 if(parent_stream(e,sid))return 0;
 if(t[1]<0)return -1;
 int64_t etable=entity_table(e);
 if(t[2]==0&&own_event_stream(e,int(t[1]))&&t[3]!=etable)return 2;
 if(t[2]==0&&parent_stream(e,int(t[1]))&&t[3]!=etable)return 3;
 if(t[2]==1&&own_event_stream(e,int(t[1]))&&t[3]!=etable&&t[3]!=e.g.table[e.q[0]])return 4;
 return -1;
}
template<class Sink> RTC_RUN int class_rows(Sink& e,int cls,int scale){
 int rows[5]={e.p.quota_parent,e.p.quota,e.p.quota_member,e.p.quota_grandparent,e.p.quota_grandchild};
 return cls<0?0:rows[cls]*scale;
}
template<class Sink> RTC_RUN int quota_rows(Sink& e,int sid,int scale){return class_rows(e,quota_class(e,sid),scale);}

template<class Sink> RTC_RUN void quota_key(Sink& e,int sid,int64_t* k){
 auto t=e.a.scratch+SW*sid;int cls=quota_class(e,sid),group=cls==2||cls==4?int(t[1]):sid;
 k[0]=cls==0?0:cls==1?1:cls==4?3:2;k[1]=0;k[2]=0;k[3]=e.a.scratch[SW*group];k[4]=t[0];
 if(cls==1||cls==2||cls==4){int64_t pop=table_population(e,e.a.scratch[SW*group+3]),epop=table_population(e,entity_table(e));
  k[1]=2*pop<=3*epop?0:1;k[2]=pop;}
}
template<class Sink> RTC_RUN bool quota_before(Sink& e,int i,int j){
 int64_t a[5],b[5];quota_key(e,i,a);quota_key(e,j,b);
 for(int x=0;x<5;++x)if(a[x]!=b[x])return a[x]<b[x];
 return i<j;
}
template<class Sink> RTC_RUN bool stream_before(Sink& e,int i,int chosen){
 if(i<0)return false;if(chosen<0)return true;
 auto t=e.a.scratch+SW*i,c=e.a.scratch+SW*chosen;

 auto left=t[10]*e.a.scratch[SW*t[12]+16]*c[11];
 auto right=c[10]*e.a.scratch[SW*c[12]+16]*t[11];
 return left<right||(left==right&&(t[0]<c[0]||(t[0]==c[0]&&i<chosen)));
}
template<class Sink> RTC_RUN int choose_stream(Sink& e){
 int chosen=-1;
 for(int i=0;i<e.s.ns;++i)if(e.a.scratch[SW*i+6]+e.a.scratch[SW*i+9]<e.a.scratch[SW*i+4]&&stream_before(e,i,chosen))chosen=i;
 return chosen;
}
template<class Sink> RTC_RUN int identity_witness(Sink& e){
 if(e.p.provenance&&e.p.forecast)for(int i=1;i<e.s.np;++i){auto v=e.a.plan+PW*i;
  if(v[0]==e.m.entity[e.q[0]]&&v[1]==0&&e.a.scratch[SW*v[2]+2]==0)return i;
 }
 return -1;
}
template<class Sink> RTC_RUN bool discovery_capped(Sink& e){
 bool capped=e.s.pruned>0;for(int i=0;i<e.s.ns;++i)capped|=e.a.scratch[SW*i+8]>0;
 return capped;
}

template<class Sink> RTC_RUN bool run_head(Sink& e){
 auto& s=e.s;auto a=e.a;auto p=e.p;auto g=e.g;auto m=e.m;auto q=e.q;int root=q[0];
 if(leader(e))setup(e);sync(e);if(s.error)return false;
 seed_ego(e);discover(e);if(s.error)return false;
 prepare_streams(e);
 int identity[2]={root,int(m.entity[root])};
 int identity_witnesses[2]={0,identity_witness(e)};
 if(admit(e,identity,p.forecast&&m.entity[root]!=root?2:1,false,identity_witnesses)<0){if(leader(e))s.error=5;sync(e);return false;}
 if(leader(e))s.seed_end=s.count;sync(e);
 if(m.kp[root+1]-m.kp[root]>1){
  for(int step=0;step<s.ns&&!s.blocked;++step){int i=a.mapping[step];auto t=a.scratch+SW*i;
    bool skip=t[17]||t[1]>=0||t[2]!=0||!t[4];sync(e);if(skip)continue;
    bool parent=false;for(auto j=m.kp[root];j<m.kp[root+1];++j)parent|=g.table[m.keys[j]]==t[3];
    if(parent){if(leader(e))t[17]=1;sync(e);bool ok=take(e,i);if(leader(e)&&!ok)s.blocked=1;sync(e);}
  }
 }
 return true;
}
template<class Sink> RTC_RUN void own_history(Sink& e){
 auto& s=e.s;auto a=e.a;auto g=e.g;int root=e.q[0];
 int prior_recent=s.recent;

 if(leader(e)&&e.settings[3]==1)s.recent=1;sync(e);

 for(int step=0,own_taken=0;step<s.ns&&!s.blocked;++step){int i=a.mapping[step];auto t=a.scratch+SW*i;
  bool own=t[19]&&t[3]==g.table[root]&&t[4];
  sync(e);if(!own)continue;
  while(!s.blocked&&t[6]<t[4]&&own_taken<OWN_HISTORY_ROWS){bool ok=take(e,i);if(leader(e)&&!ok)s.blocked=1;sync(e);++own_taken;}
 }
 if(leader(e))s.recent=prior_recent;sync(e);
}

template<class Sink> RTC_RUN int next_quota_stream(Sink& e,int previous,int scale){
 int best=-1;
 for(int i=0;i<e.s.ns;++i){if(!quota_rows(e,i,scale))continue;
  if(previous>=0&&!quota_before(e,previous,i))continue;
  if(best<0||quota_before(e,i,best))best=i;}
 return best;
}
template<class Sink> RTC_RUN void quota_tier(Sink& e,int scale){
 auto& s=e.s;auto a=e.a;auto p=e.p;
 int rounds=maximum(maximum(maximum(p.quota,p.quota_member),maximum(p.quota_parent,p.quota_grandparent)),p.quota_grandchild)*scale;
 if(leader(e)){s.blocked=0;s.recent=1;}sync(e);
 for(int round=0;round<rounds&&!s.blocked;++round)
  for(int step=0,i=-1;step<s.ns&&!s.blocked;++step){
   if(p.quota_order){i=next_quota_stream(e,i,scale);if(i<0)break;}else i=int(a.mapping[step]);
   auto t=a.scratch+SW*i;
   bool go=round<quota_rows(e,i,scale)&&t[6]<=round&&t[6]<t[4];sync(e);if(!go)continue;
   bool ok=take(e,i);if(leader(e)){if(ok)t[17]=1;else s.blocked=1;}sync(e);
  }
 if(leader(e)){s.blocked=0;s.recent=0;}sync(e);
}
template<class Sink> RTC_RUN void run_tail(Sink& e){
 auto& s=e.s;auto a=e.a;auto p=e.p;int root=e.q[0];

 for(int step=0;step<s.ns&&!s.blocked;++step){int chosen=a.mapping[step];
  auto t=a.scratch+SW*chosen;bool skip=t[17];sync(e);if(skip)continue;if(leader(e))t[17]=1;sync(e);
  if(t[4]){bool ok=take(e,chosen);if(leader(e)&&!ok)s.blocked=1;sync(e);}
 }
 while(!s.blocked&&!s.error){int chosen=choose_stream(e);
  if(chosen<0)break;bool ok=take(e,chosen);if(leader(e)&&!ok)s.blocked=1;sync(e);
 }

 bool capped=discovery_capped(e);
 if(!s.blocked&&!capped&&p.filler){
  for(int r=0;r<p.N;++r)if(r!=root&&admit(e,&r,1,true)<0)break;
 }
}
EP_HD inline bool quota_tiers(Policy p){return p.quota||p.quota_member||p.quota_parent||p.quota_grandparent||p.quota_grandchild;}
EP_HD inline bool general_tiers(Policy p){return quota_tiers(p)||p.own_labels||p.seed_percent||p.own_recent;}
template<class Sink> RTC_RUN void run(Sink& e){
 if(!run_head(e))return;
 own_history(e);run_tail(e);
}
template<class Sink> RTC_RUN void begin_extension(Sink& e){
 for(int i=lane(e);i<group_capacity(e.p);i+=lanes(e))e.a.table_counts[i]=0;
 if(leader(e))e.s.extension=1;sync(e);
}
template<class Sink> RTC_RUN void extend_paths(Sink& e){

 if(leader(e))e.s.blocked=0;sync(e);
 while(!e.s.error&&e.s.count<e.settings[0]){
  int chosen=choose_stream(e);if(chosen<0)break;
  bool ok=take(e,chosen);if(leader(e)&&!ok)e.s.blocked=1;sync(e);
 }
}
template<class Sink> RTC_RUN void receipt_totals(Sink& e){
 auto& s=e.s;auto r=e.a.receipt;
 for(int i=0;i<s.ns;++i){auto t=e.a.scratch+SW*i;
  r[23]+=t[4];r[24]+=t[5];r[25]+=t[6];r[26]+=t[7];
  r[27]=t[8]>INT64_MAX-r[27]?INT64_MAX:r[27]+t[8];
 }
}
template<class Sink> RTC_RUN void finish(Sink& e){
 auto& s=e.s;auto r=e.a.receipt;
 if(leader(e)){r[0]=e.settings[0];r[1]=s.count;r[2]=s.rows;r[3]=s.peers;r[4]=s.kenc;r[5]=s.targets;
 r[7]=s.peers;r[8]=s.history;r[9]=s.text;r[10]=s.filler;
 r[11]=s.seed_end;r[12]=s.seed_end+s.own_cells;r[13]=s.count-s.filler;
 r[14]=s.rw_scanned;r[15]=s.rw_truncated;r[16]=s.rw_graph_exhausted;
 r[17]=2;r[18]=0;r[19]=0;r[20]=0;r[21]=0;
 r[22]=s.ns;r[23]=0;r[24]=0;r[25]=0;r[26]=0;r[27]=s.pruned;r[28]=s.blocked?r[0]-s.count:0;
 r[30]=s.total;r[31]=s.pruned;r[32]=s.frontier;r[33]=s.saturated;r[34]=s.ego_rows;r[35]=s.ego_cap;
 }
 for(int i=lane(e);i<s.ns;i+=lanes(e)){auto t=e.a.scratch+SW*i;t[9]=t[4]-t[6];
  for(int j=0;j<12;++j)r[STREAM_OFFSET+STREAM_WIDTH*i+j]=t[j];r[STREAM_OFFSET+STREAM_WIDTH*i+12]=t[19];
 }
 sync(e);
 receipt_totals(e);
 if(leader(e)){
 r[6]=s.error?-s.error:(s.count==r[0]?0:s.blocked?2:r[27]?3:1);
 r[29]=!s.blocked&&!s.error&&!r[27];

 if(e.p.core&&e.settings[0]>e.p.core){r[17]=3;r[29]=0;}
 }
 sync(e);
}
}
