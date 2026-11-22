#include <torch/extension.h>
#include <climits>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>
#include "rt_composition.h"

Tensors rt_compose_v2_cuda(Tensors,Tensors,Tensors,Tensor,Tensor,Tensors,std::vector<int64_t>);
namespace rtc {
struct CPU {
 RTGraph g;Meta m;Policy p;Work a;State& s;const int64_t *q,*settings;RTIndex32 fp,fd,pp,pd;
};
}
Tensors rt_compose_v2(Tensors g,Tensors d,Tensors m,Tensor q,Tensor settings,Tensors w,std::vector<int64_t> v){
 TORCH_CHECK(g.size()==11&&d.size()==6&&(m.size()>=19&&m.size()<=22)&&w.size()==15&&v.size()==rtc::POLICY_SIZE,"composition ABI mismatch; rebuild remotely");
 auto device=q.device();
 TORCH_CHECK(q.dim()==2&&q.size(1)==5&&q.is_contiguous()&&q.scalar_type()==at::kLong,"composition requests");
 TORCH_CHECK(settings.sizes()==at::IntArrayRef({q.size(0),5})&&settings.is_contiguous()&&settings.scalar_type()==at::kLong&&settings.device()==device,"composition settings");
 for(auto list:{g,m})for(auto x:list)TORCH_CHECK(x.dim()==1&&x.is_contiguous()&&rt_index_type(x)&&x.device()==device,"composition metadata dtype/device");
 for(int i=0;i<6;++i)TORCH_CHECK(d[i].dim()==1&&d[i].is_contiguous()&&d[i].device()==device&&rt_index_type(d[i]),"composition CSR dtype/device");
 rtc::Policy p{int(v[0]),int(v[1]),int(v[2]),int(v[3]),int(v[4]),int(v[5]),int(v[6]),int(v[7]),int(v[8]),int(v[9]),int(v[10]),int(v[11]),int(v[12]),int(v[13]),int(v[14]),int(v[15]),int(v[16]),int(v[17]),int(v[18]),int(v[19]),int(v[20]),int(v[21]),int(v[22]),int(v[23]),int(v[24]),int(v[25]),int(v[26]),int(v[27]),int(v[28]),int(v[29]),int(v[30]),int(v[31]),int(v[32]),int(v[33]),int(v[34]),int(v[35]),int(v[36]),int(v[37]),int(v[38]),int(v[39]),int(v[40]),int(v[41]),int(v[42]),int(v[43]),int(v[44]),int(v[45]),int(v[46]),int(v[47]),int(v[48]),int(v[49]),int(v[50]),int(v[51]),int(v[52]),int(v[53]),int(v[54]),int(v[55])};
 TORCH_CHECK(p.core>=0&&p.core<=p.L&&(!p.core||p.K<=64),"protected RW policy bounds");
 TORCH_CHECK((p.mixed==0||p.mixed==1)&&(!p.mixed||p.core>0)&&p.rw_percent>=0&&p.rw_percent<=100&&p.time_unit>0&&p.rw_scan_budget>0,"mixed extension policy bounds");
 TORCH_CHECK(int64_t(p.T)*(p.mixed?p.strata+1:1)<INT_MAX,"mixed group workspace overflow");
 TORCH_CHECK(p.provenance==0||p.provenance==1,"composition provenance capability bounds");
 TORCH_CHECK(p.own_recent>=0&&p.own_recent<=4&&((p.own_recent!=2&&p.own_recent!=3)||p.T<=64),"own recency policy bounds");
 TORCH_CHECK(p.canonical==0||p.canonical==1,"canonical order policy bounds");
 TORCH_CHECK((p.scale_discovery==0||p.scale_discovery==1)&&(p.weight_policy==0||p.weight_policy==1),"composition discovery/weight policy bounds");
 TORCH_CHECK(p.truncate==0||p.truncate==1,"composition seed policy bounds");
 TORCH_CHECK(p.reserve>=0&&2LL*p.reserve<=p.seed_rows,"label_pool_reserve must lie in [0, frontier_rows/2]");
 TORCH_CHECK(p.seed_percent>=0&&p.seed_percent<=90&&p.seed_local>=0&&p.seed_local<=64&&(p.seed_balance==0||p.seed_balance==1)
             &&(p.seed_percent||(!p.seed_local&&!p.seed_balance))&&(!p.seed_percent||p.K<=64),"seed tier policy bounds");
 TORCH_CHECK(p.quota>=0&&p.quota<=64&&p.quota_member>=0&&p.quota_member<=64&&p.quota_share>=0&&p.quota_share<=100
             &&(rtc::quota_tiers(p)||!p.quota_share),"path quota policy bounds");
 TORCH_CHECK(p.own_labels>=0&&p.own_labels<=256&&(p.seed_fill==0||p.seed_fill==1)
             &&(p.seed_hide==0||p.seed_hide==1)&&(p.seed_local_policy==0||p.seed_local_policy==1),"general sampler tier policy bounds");
 TORCH_CHECK(p.seed_fanout>=1&&p.seed_fanout<=4096&&(p.stream_fill==0||p.stream_fill==1),"shared-route fanout / stream fill policy bounds");
 TORCH_CHECK(p.seed_rank>=0&&p.seed_rank<=2&&p.seed_hop_cap>=1&&p.seed_hop_cap<=2048,"program seed ranking bounds");
 TORCH_CHECK((p.seed_program_mature==0||p.seed_program_mature==1)&&(!p.seed_program_mature||p.seed_rank==2),"program maturity policy bounds");
 TORCH_CHECK(p.peer_ext>=0&&p.peer_ext<=100&&p.peer_ext_rows>=0&&p.peer_ext_rows<=64&&(p.peer_ext||!p.peer_ext_rows)
             &&(!p.peer_ext||(p.mixed&&p.seed_percent&&p.peer_ext+p.rw_percent<=100)),"peer extension policy bounds");
 TORCH_CHECK(p.quota_parent>=0&&p.quota_parent<=64&&p.quota_grandparent>=0&&p.quota_grandparent<=64
             &&p.quota_grandchild>=0&&p.quota_grandchild<=64&&(p.quota_order==0||p.quota_order==1),"path quota class bounds");
 TORCH_CHECK(p.L<=rtc::MAX_SEQ_LEN,"path trace exceeds MAX_SEQ_LEN");
 for(auto x:v)TORCH_CHECK(x>=-1&&x<INT_MAX,"composition policy overflow");
 TORCH_CHECK(p.W>0&&p.K>0&&p.W<=1024/p.K&&p.C>0&&p.C<=p.W*p.K&&p.L>0&&p.F>0&&p.h>=0&&p.h<=64&&p.T>0&&p.strata>0&&p.strata<=32&&p.filler>=0&&p.filler<=(p.history>=2?100:20),"composition policy bounds");
 TORCH_CHECK(p.history>=0&&p.history<=3&&!(p.history>=2&&p.forecast),"composition label policy bounds");
 TORCH_CHECK(p.F<=32768&&p.text>=0&&p.text<=2&&p.probes>0&&p.probes<=64&&p.card>=0&&p.rate>0&&p.rate<=1024&&p.kmax>=0,"composition typed bounds");
 TORCH_CHECK(p.S>0&&p.S<=4096&&p.seed_hops>=0&&p.seed_hops<=p.h&&p.seed_rows>0&&p.seed_rows<=p.F,"composition schema/ego bounds");
 TORCH_CHECK(p.H>=4LL*p.F&&(p.H&(p.H-1))==0&&int64_t(p.E)>=4LL*p.L&&(p.E&(p.E-1))==0,"composition hash capacity");
 auto B=w[0].size(0),N=g[3].numel(),M=g[7].numel();
 if(m.size()>=21)TORCH_CHECK(m[20].numel()<=N,"composition label index alignment");
 if(m.size()==22)TORCH_CHECK(m[21].numel()==d[4].numel(),"composition recency index alignment");
 TORCH_CHECK(p.N==N,"composition row count mismatch");
 TORCH_CHECK(q.size(0)<=B&&N>0&&N<INT_MAX&&M>0&&M<INT_MAX&&B*p.E<INT_MAX&&B*p.H<INT_MAX&&B*p.C*6<INT_MAX&&B*p.W*p.K+255<INT_MAX,"composition address overflow");
 TORCH_CHECK(g[6].numel()==N+1&&g[4].numel()==N&&g[5].numel()==N,"composition row alignment");
 for(int i:{8,9,10})TORCH_CHECK(g[i].numel()==M,"composition cell alignment");
 for(int i:{0,1})TORCH_CHECK(m[i].numel()==M,"composition feature alignment");
 for(int i:{2,3,4})TORCH_CHECK(m[i].numel()==N,"composition row metadata alignment");
 TORCH_CHECK(m[5].numel()==N+1&&m[7].numel()==p.T&&m[8].numel()==N+1&&m[9].numel()==m[10].numel()+1&&m[11].numel()==m[12].numel(),"composition CSR alignment");
 TORCH_CHECK(m[13].numel()==N+1,"composition entity key CSR alignment");
 TORCH_CHECK(m.size()==19||m[19].numel()==0||m[19].numel()==N,"composition exact source clock alignment");
 TORCH_CHECK(m[15].numel()==p.T&&m[16].numel()==p.T&&m[17].numel()==p.T&&m[18].numel()==1,"composition task stream mask alignment");
 TORCH_CHECK(d[0].numel()==N+1&&d[3].numel()==N+1,"composition directed row alignment");
 TORCH_CHECK(w[12].numel()>=int64_t(p.T)*rtc::SCHEMA_HEADER&&w[12].numel()<INT_MAX,"composition schema alignment");

 std::vector<std::vector<int64_t>> shapes={{B,p.W,p.K},{B,p.C},{B,p.C},{B,p.L},{B,p.L},{B,rtc::receipt_size(p)},{B,p.F,rtc::PW},{B,rtc::group_capacity(p)},{B,p.H},{B,p.E},{B,p.E},{B,p.S,rtc::SW},{w[12].numel()},{B,p.S},{B,rtc::seed_work_size(p)}};
 for(int i=0;i<15;++i){
  TORCH_CHECK(w[i].sizes().vec()==shapes[i]&&w[i].is_contiguous()&&w[i].scalar_type()==at::kLong&&w[i].device()==device,"composition workspace alignment at ",i);
  TORCH_CHECK(w[i].numel()+255<INT_MAX,"composition workspace address overflow at ",i);
 }
#ifdef WITH_CUDA
 if(q.is_cuda())return rt_compose_v2_cuda(g,d,m,q,settings,w,v);
#endif
 TORCH_CHECK(!q.is_cuda(),"composition CUDA kernel unavailable; rebuild remotely");

 if(p.weight_policy){auto schema=w[12].data_ptr<int64_t>();int64_t n=w[12].numel();
  for(int t=0;t<p.T;++t){auto header=schema+rtc::SCHEMA_HEADER*t;
   TORCH_CHECK(header[1]>=0&&header[2]>=0&&header[1]+rtc::SCHEMA_WIDTH*header[2]<=n,"composition schema header out of range");
   for(int64_t i=0;i<header[2];++i){int64_t scale=schema[header[1]+rtc::SCHEMA_WIDTH*i+5];
    TORCH_CHECK(scale>=1&&scale<=(int64_t(1)<<20),"composition stream weight scale outside [1, 1<<20]: ",scale);}}}
 RTGraph graph{rt_index(g[0]),rt_index(g[1]),rt_index(g[2]),rt_index(g[3]),rt_index(g[4]),rt_index(g[5]),rt_index(g[6]),rt_index(g[7]),rt_index(g[8]),rt_index(g[9]),rt_index(g[10])};
 rtc::Meta meta{rt_index(m[0]),rt_index(m[1]),rt_index(m[2]),rt_index(m[3]),rt_index(m[4]),rt_index(m[5]),rt_index(m[6]),rt_index(m[7]),rt_index(m[8]),rt_index(m[9]),rt_index(m[10]),rt_index(m[11]),rt_index(m[12]),rt_index(m[13]),rt_index(m[14]),rt_index(m[15]),rt_index(m[16]),rt_index(m[17]),rt_index(m[18]),rt_address(d[0]),rt_address(d[1]),rt_index((m.size()>=20&&m[19].numel()>0)?m[19]:g[4]),m.size()>=20&&m[19].numel()>0};
 if(m.size()>=21){meta.pool=rt_index(m[20]);meta.pool_size=int(m[20].numel());}
 if(m.size()==22)meta.recent=rt_index(m[21]);
 w[0].fill_(-1);w[1].fill_(-1);w[2].zero_();w[3].fill_(-1);w[4].zero_();w[9].fill_(-1);w[10].fill_(-1);

 const int64_t episodes=q.size(0);
 std::vector<std::string> errors(episodes);
 auto episode=[&](int64_t b){auto r=q.data_ptr<int64_t>()+5*b,s=settings.data_ptr<int64_t>()+5*b;
  TORCH_CHECK(r[0]>=0&&r[0]<N&&s[0]>0&&s[0]<=p.L&&s[1]>0&&s[1]<INT_MAX&&s[4]>0&&s[4]<=p.L,"composition invalid request");
  rtc::Work a{w[3].data_ptr<int64_t>()+b*p.L,w[4].data_ptr<int64_t>()+b*p.L,w[5].data_ptr<int64_t>()+b*rtc::receipt_size(p),w[6].data_ptr<int64_t>()+b*p.F*rtc::PW,w[7].data_ptr<int64_t>()+b*rtc::group_capacity(p),w[8].data_ptr<int64_t>()+b*p.H,w[9].data_ptr<int64_t>()+b*p.E,w[10].data_ptr<int64_t>()+b*p.E,w[11].data_ptr<int64_t>()+b*p.S*rtc::SW,w[1].data_ptr<int64_t>()+b*p.C,w[2].data_ptr<int64_t>()+b*p.C,w[12].data_ptr<int64_t>(),w[13].data_ptr<int64_t>()+b*p.S,w[14].data_ptr<int64_t>()+b*rtc::seed_work_size(p)};

  int64_t core_total=p.core?std::min(s[0],int64_t(p.core)):s[0];
  int64_t chosen[5]={core_total,s[1],s[2],s[3],s[4]};
  rtc::State state{};rtc::CPU engine{graph,meta,p,a,state,r,chosen,rt_address(d[0]),rt_address(d[1]),rt_address(d[3]),rt_address(d[4])};
  int64_t* visits=w[0].data_ptr<int64_t>()+b*p.W*p.K;
  if(!rtc::general_tiers(p))rtc::run(engine);
  else if(rtc::run_head(engine)){
   int scale=int(std::max<int64_t>(1,core_total/1024));
   rtc::RWSerial serial{graph,meta,p,a,state,r,chosen,engine.fp,engine.fd,engine.pp,engine.pd};
   if(p.own_recent)rtc::own_recent_tier(serial);
   else if(p.own_labels)rtc::own_label_tier(serial,scale);else rtc::own_history(engine);
   if(rtc::quota_tiers(p)){
    chosen[0]=p.quota_share?std::max<int64_t>(state.count,core_total*p.quota_share/100):core_total;
    rtc::quota_tier(engine,scale);chosen[0]=core_total;
   }
   if(p.seed_percent){
    chosen[0]=std::min<int64_t>(core_total,state.count+core_total*p.seed_percent/100);
    rtc::seed_tier(serial,visits);chosen[0]=core_total;
   }

   state.blocked=0;if(p.stream_fill)rtc::run_tail(engine);else state.blocked=1;
  }
  if(p.core&&s[0]>p.core&&!state.error){chosen[0]=s[0];
   rtc::RWSerial rw{graph,meta,p,a,state,r,chosen,engine.fp,engine.fd,engine.pp,engine.pd};
   if(p.mixed){rtc::begin_extension(engine);
    if(p.peer_ext){chosen[0]=state.count+(s[0]-p.core)*p.peer_ext/100;rtc::peer_extension(rw);chosen[0]=s[0];}
    chosen[0]=state.count+(s[0]-p.core)*p.rw_percent/100;}
   if(!p.mixed||p.rw_percent){
    if(state.walked!=1){if(state.walked==2)std::fill(visits,visits+p.W*p.K,int64_t(-1));rtc::generate_rw(rw,visits);}
    rtc::extend_rw(rw,visits);
   }
   if(p.mixed){chosen[0]=s[0];rtc::extend_paths(engine);}
  }
  rtc::finish(engine);
  TORCH_CHECK(state.error!=6&&state.error!=7,"composition path capacity/collision failure; code=",state.error);
  TORCH_CHECK(state.error!=1&&state.error!=5,"composition seed budget failure: complete or protected seed cells exceed L; code=",state.error);
  TORCH_CHECK(state.error!=2,"composition frontier failure: ego row plan exceeds frontier_rows; code=",state.error);
  TORCH_CHECK(state.error!=3&&state.error!=4,"composition identity failure: missing focal target or forecast identity; code=",state.error);
  TORCH_CHECK(state.error!=9,"query table is masked out for this task");
  TORCH_CHECK(state.error!=10,"composition stream weight scale outside [1, 1<<20]; code=",state.error);
  TORCH_CHECK(state.error==0,"composition invalid admitted schema or witness; code=",state.error);
 };
 auto run_range=[&](int64_t begin,int64_t end){
  for(int64_t b=begin;b<end;++b){
   try{episode(b);}
   catch(const c10::Error& error){errors[b]=error.what_without_backtrace();}
   catch(const std::exception& error){errors[b]=error.what();}
  }
 };
 const int64_t workers=std::max<int64_t>(1,std::min<int64_t>(episodes,at::get_num_threads()));
 if(workers<=1){run_range(0,episodes);}
 else{
  std::vector<std::thread> pool;pool.reserve(workers);
  for(int64_t t=0;t<workers;++t)pool.emplace_back(run_range,episodes*t/workers,episodes*(t+1)/workers);
  for(auto& thread:pool)thread.join();
 }
 for(int64_t b=0;b<episodes;++b)TORCH_CHECK(errors[b].empty(),errors[b]);
 return {w[0],w[1],w[2],w[3],w[4],w[5]};
}
