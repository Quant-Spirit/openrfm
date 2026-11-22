#pragma once
#include "episodes.h"

struct RTIndex {
    const void* data; int width;
    EP_HD RTIndex operator+(int64_t i) const { return {static_cast<const uint8_t*>(data)+i*width,width}; }
    EP_HD RTIndex operator-(int64_t i) const { return *this + (-i); }
    EP_HD int64_t operator[](int64_t i) const {
        switch(width) {
            case 1: return static_cast<const uint8_t*>(data)[i];
            case 2: return static_cast<const uint16_t*>(data)[i];
            case 4: return static_cast<const int32_t*>(data)[i];
            default: return static_cast<const int64_t*>(data)[i];
        }
    }
};
inline bool rt_index_type(const Tensor& x) {
    auto t=x.scalar_type();
    return t==at::kByte||t==at::kUInt16||t==at::kInt||t==at::kLong;
}
inline RTIndex rt_index(const Tensor& x) {
    TORCH_CHECK(rt_index_type(x), "RT resident index has unsupported dtype");
    return {x.data_ptr(),int(x.element_size())};
}
struct RTIndex32 : RTIndex {
    EP_HD int32_t operator[](int64_t i) const { return int32_t(RTIndex::operator[](i)); }
    EP_HD RTIndex32 operator+(int64_t i) const { return {RTIndex::operator+(i)}; }
    EP_HD RTIndex32 operator-(int64_t i) const { return *this + (-i); }
};
inline RTIndex32 rt_address(const Tensor& x) { return {rt_index(x)}; }
#ifdef __CUDACC__
__device__ inline int64_t __ldg(RTIndex x) { return x[0]; }
__device__ inline int32_t __ldg(RTIndex32 x) { return x[0]; }
#endif
struct RTGraph {
    RTIndex ptr,dst,time,table,rowtime,task,cp,col,avail,label,drop;
};
struct RTPolicy {int64_t W,K,C,L,hops,history,legacy;};
EP_HD inline uint64_t rt_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
EP_HD inline bool rt_time_ok(int64_t time,int64_t cut,bool strict) {
    return strict ? time < cut : time <= cut;
}
EP_HD inline int64_t rt_end(RTGraph g,int64_t row,int64_t cut,bool strict) {
    int64_t lo=g.ptr[row],hi=g.ptr[row+1];
    while(lo<hi){int64_t m=lo+(hi-lo)/2;
        if(rt_time_ok(g.time[m],cut,strict))lo=m+1;else hi=m;}
    return lo;
}
EP_HD inline bool rt_row_ok(RTGraph g,int64_t row,int64_t root) {
    return row==root || !g.task[row] || (g.task[row]==1&&g.table[row]==g.table[root]);
}

EP_HD inline bool rt_row_ok_forecast(RTGraph g,int64_t row,int64_t root,bool forecast) {
    return rt_row_ok(g,row,root)
        || (forecast && g.task[row]!=0 && g.table[row]==g.table[root]);
}
EP_HD inline void rt_walk(RTGraph g,const int64_t* q,RTPolicy p,
                          int64_t walk,int64_t* out) {
    int64_t root=q[0],row=root;
    for(int64_t k=0;k<p.K;++k){
        int64_t lo=g.ptr[row],end=rt_end(g,row,q[1],q[4]);

        int64_t n=0,next=-1;
        for(int64_t j=lo;j<end;++j)if(rt_row_ok(g,g.dst[j],root)){
            ++n;
            uint64_t key=rt_mix(uint64_t(q[3])^rt_mix(walk)^rt_mix(k+p.K)^rt_mix(j));
            if(key%uint64_t(n)==0)next=g.dst[j];
        }

        row=next<0?root:next;
        out[k]=next>=0&&row!=root&&g.table[row]==g.table[root]?row:-1;
    }
}
EP_HD inline int64_t rt_bound(const int64_t* a,int64_t n,int64_t x,bool upper) {
    int64_t lo=0,hi=n;while(lo<hi){int64_t m=lo+(hi-lo)/2;
        if(a[m]<x||(upper&&a[m]==x))lo=m+1;else hi=m;}return lo;
}

EP_HD inline void rt_rank(RTGraph g,const int64_t* sorted,int64_t n,int64_t i,
                          int64_t rho,int64_t C,int64_t* rows,int64_t* scores) {
    int64_t x=sorted[i];if(x<0||(i&&sorted[i-1]==x))return;
    int64_t count=rt_bound(sorted,n,x,true)-i,rank=0;
    for(int64_t j=0;j<n;){int64_t y=sorted[j],end=rt_bound(sorted,n,y,true);
        if(y>=0){int64_t c=end-j;
            bool ahead=c>count||(c==count&&((rho&&g.rowtime[y]>g.rowtime[x])||
                ((!rho||g.rowtime[y]==g.rowtime[x])&&y<x)));
            rank+=ahead;}
        j=end;
    }
    if(rank<C){rows[rank]=x;scores[rank]=count;}
}
EP_HD inline int64_t rt_slot(int64_t row,int64_t* hash,int64_t H) {
    int64_t pos=rt_mix(row)&(H-1);
    while(hash[pos]!=-1&&hash[pos]!=row)pos=(pos+1)&(H-1);
    return pos;
}
EP_HD inline int64_t rt_local_slot(int64_t row,int64_t* ids,int64_t* epochs,int64_t H,int64_t epoch) {
    int64_t pos=rt_mix(row)&(H-1);
    while(epochs[pos]==epoch&&ids[pos]!=row)pos=(pos+1)&(H-1);
    return pos;
}
EP_HD inline bool rt_cell_ok(RTGraph g,int64_t c,int64_t row,const int64_t* q,RTPolicy p) {
    bool focal=row==q[0]&&g.col[c]==q[2];
    if(focal)return true;
    if(g.drop[c])return false;
    bool label=g.label[c]||g.col[c]==q[2];
    if(label){
        if(!p.history||row==q[0])return false;

        return g.avail[c]<q[1];
    }
    return g.avail[c]<=q[1];
}

EP_HD inline void rt_bfs(RTGraph g,const int64_t* q,const int64_t* settings,
        RTPolicy p,const int64_t* candidates,int64_t* cells,int64_t* roles,
        int64_t* receipt,int64_t* queue,int64_t* hash,int64_t H) {
    int64_t L=settings[0],ell=p.legacy?L:settings[1],width=settings[2];
    int64_t used=0,rows=0,donors=0,labels=0,targets=0,reason=0;

    int64_t root_slot=rt_slot(q[0],hash,H);
    for(int64_t c=g.cp[q[0]];c<g.cp[q[0]+1];++c)if(g.col[c]==q[2]){
        cells[used]=c;roles[used]=q[0];++used;++targets;
    }
    if(used){hash[root_slot]=q[0];hash[H+root_slot]=g.cp[q[0]];hash[3*H+root_slot]=0;rows=1;}

    for(int64_t pass=0;pass<=L&&used<L;++pass){int64_t before=used;
        for(int64_t launch=0;launch<=(p.legacy?0:p.C)&&used<L;++launch){
            int64_t start=launch==0?q[0]:candidates[launch-1];if(start<0)continue;
            int64_t* visited=hash+2*H;
            int64_t* epochs=hash+4*H;int64_t epoch=pass*(p.C+1)+launch;
            int64_t first=rt_local_slot(start,visited,epochs,H,epoch);
            visited[first]=start;epochs[first]=epoch;
            int64_t head=0,tail=1,spent=(pass==0&&launch==0)?targets:0;
            queue[0]=start;queue[1]=0;queue[2]=launch==0?0:2;
            while(head<tail&&spent<ell&&used<L){
                int64_t row=queue[4*head],depth=queue[4*head+1],role=queue[4*head+2];++head;
                int64_t slot=rt_slot(row,hash,H);
                int64_t cursor=hash[slot]==row?hash[H+slot]:g.cp[row];
                if(hash[slot]==row&&role==1&&row!=q[0])hash[3*H+slot]=1;
                for(;cursor<g.cp[row+1]&&spent<ell&&used<L;++cursor){
                    if(row==q[0]&&g.col[cursor]==q[2])continue;
                    if(!rt_cell_ok(g,cursor,row,q,p))continue;
                    if(hash[slot]==-1){hash[slot]=row;++rows;hash[3*H+slot]=role;donors+=role==2;}
                    cells[used]=cursor;roles[used]=row;++used;++spent;
                    labels+=g.label[cursor]||g.col[cursor]==q[2];
                }
                if(hash[slot]==row)hash[H+slot]=cursor;
                if(spent==ell||used==L)break;
                if(p.hops&&depth>=p.hops)continue;
                int64_t lo=g.ptr[row],end=rt_end(g,row,q[1],q[4]),n=end-lo,accepted=0;
                int64_t offset=n?rt_mix(uint64_t(q[3])^rt_mix(row))%uint64_t(n):0;
                for(int64_t j=0;j<n&&accepted<width;++j){int64_t child=g.dst[lo+(offset+j)%n];
                    if(!rt_row_ok(g,child,q[0]))continue;
                    int64_t local=rt_local_slot(child,visited,epochs,H,epoch);if(epochs[local]==epoch)continue;
                    if(tail>=2*L){reason=2;break;}
                    visited[local]=child;epochs[local]=epoch;queue[4*tail]=child;queue[4*tail+1]=depth+1;
                    queue[4*tail+2]=child==q[0]?0:1;++tail;++accepted;
                }
            }
        }
        if(used==before)break;
    }
    for(int64_t i=0;i<used;++i)roles[i]=hash[3*H+rt_slot(roles[i],hash,H)];
    receipt[0]=L;receipt[1]=used;receipt[2]=rows;receipt[3]=donors;
    receipt[4]=labels;receipt[5]=targets;receipt[6]=used==L?0:(reason?reason:1);
}

Tensors rt_retrieve(Tensors graph,Tensor requests,Tensor settings,
                    int64_t W,int64_t K,int64_t C,int64_t L,int64_t hops,int64_t history,int64_t legacy);
#ifdef WITH_CUDA
Tensors rt_retrieve_cuda(Tensors,Tensor,Tensor,RTPolicy);
#endif
