__kernel void reduce(__global float *data, __local float *tmp)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    tmp[lid] = data[gid]
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = get_local_size(0) / 2; i > 0; i >= 1){
        if(lid < i){
            tmp[lid] += tmp[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0){
        data[get_group_id(0)] = tmp[0];
    }
}