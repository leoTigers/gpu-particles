#include "kernel.cuh"

#define G (float)1e-15
#define DRAG (float)0.5f
#define MOUSE_POWER 500000

__global__ void computeForces(Particle* p, param *par, float *VC)
{
    int indice = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (indice >= PARTICLE_TOTAL)
        return;
    
    float dist = fmaxf(
        sqrtf(powf(VC[2*indice] - par->mousePos.x, 2.f) + powf(VC[2*indice+1] - par->mousePos.y, 2.f)),
        0.0001);

    p[indice].m_force.x = (par->mousePos.x - VC[2 * indice]) * (float)(par->LMB * MOUSE_POWER / powf(dist + 10, 2.f));
    p[indice].m_force.y = (par->mousePos.y - VC[2 * indice+1]) * (float)(par->LMB * MOUSE_POWER / powf(dist + 10, 2.f));
    


    /* for all particles between themselves 
    float Vx, Vy, dist, norm;
    int i;
    p[indice].m_force.x = 0.f;
    p[indice].m_force.y = 0.f;

    for (i = 0; i < PARTICLE_TOTAL; i++)
    {
        if (i == indice)
            continue;
        Vx = p[i].m_pos.x - p[indice].m_pos.x;
        Vy = p[i].m_pos.y - p[indice].m_pos.y;
        //printf("id:%d\t%f %f\n", indice, Vx, Vy);
        norm = Vx * Vx + Vy * Vy;
        dist = distance(p, indice, i);
        //printf("dist %d %d = %f", indice, i, dist);
        Vx *=  G * p[i].m_mass * p[indice].m_mass / (dist * dist * dist);
        Vy *=  G * p[i].m_mass * p[indice].m_mass / (dist * dist * dist);

        //printf("=======>id:%d\t%f %f\n", indice, Vx, Vy);
        p[indice].m_force.x += Vx;
        p[indice].m_force.y += Vy;
        //p[i].m_force.x -= Vx;
        //p[i].m_force.y -= Vy;
    }
    /**/
    p[indice].m_force.x -= p[indice].m_speed.x * DRAG;
    p[indice].m_force.y -= p[indice].m_speed.y * DRAG;
    /**/
}

__global__ void computeColors(Particle* p, param* par, unsigned char* colors)
{
    int indice = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (indice >= PARTICLE_TOTAL)
        return;

    float start1 = 0;
    float stop1 = 200, stop2 = 1536;

    float value = sqrtf(p[indice].m_speed.x * p[indice].m_speed.x + p[indice].m_speed.y * p[indice].m_speed.y);
    value = value > stop1 ? stop1 : value;

    float ratio =  (stop2 - stop1) * ((value - start1) / (stop1 - start1));
    colors[3 * indice] = 0;
    colors[3 * indice + 1] = 0;
    colors[3 * indice + 2] = 0;

    if (ratio < 256)
    {
        colors[3 * indice] = 255;
        colors[3 * indice + 1] = ratio;
    }
    else if (ratio < 512)
    {
        colors[3 * indice] = 512 - ratio;
        colors[3 * indice + 1] = 255;
    }
    else if (ratio < 768)
    {
        colors[3 * indice + 1] = 255;
        colors[3 * indice + 2] = ratio - 512;
    }
    else if (ratio < 1024)
    {
        colors[3 * indice + 1] = 1024 - ratio;
        colors[3 * indice + 2] = 255;
    }
    else if (ratio < 1280)
    {
        colors[3 * indice + 2] = 255;
        colors[3 * indice] = ratio - 1024;
    }
    else
    {
        colors[3 * indice + 2] = 1536 - ratio;
        colors[3 * indice] = 255;
    }
}

__global__ void computePositions(Particle* p, param* par, float* VC)
{
    int indice = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (indice >= PARTICLE_TOTAL)
        return;
    float pos_x = VC[2 * indice];
    float pos_y = VC[2 * indice + 1];

    VC[2 * indice] = VC[2 * indice] + p[indice].m_speed.x * par->dt + .5f * p[indice].m_force.x / p[indice].m_mass * par->dt * par->dt;
    VC[2 * indice + 1] = VC[2 * indice + 1] + p[indice].m_speed.y * par->dt + .5f * p[indice].m_force.y / p[indice].m_mass * par->dt * par->dt;

    p[indice].m_speed.x = (VC[2 * indice] - pos_x) / par->dt;
    p[indice].m_speed.y = (VC[2 * indice + 1] - pos_y) / par->dt;
}



cudaError_t computeParticles(Particle* particles, Particle * d_particles, param *p, float* VC, float* d_VC, 
    unsigned char* colors, unsigned char* d_colors)
{
    cudaError_t error;
    error = cudaSetDevice(0);
    if (error != cudaSuccess)
        printf("GPUassert: %s \n", cudaGetErrorString(error));
    
    
    computeForces << <BLOCK_COUNT, THREAD_BLOCK_SIZE >> > (d_particles, p, d_VC);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        printf("GPUassert: %s \n", cudaGetErrorString(error));
    computePositions << <BLOCK_COUNT, THREAD_BLOCK_SIZE >> > (d_particles, p, d_VC);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        printf("GPUassert: %s \n", cudaGetErrorString(error));
    computeColors << <BLOCK_COUNT, THREAD_BLOCK_SIZE >> > (d_particles, p, d_colors);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        printf("GPUassert: %s \n", cudaGetErrorString(error));
    
    /*
    error = cudaMemcpy(particles, d_particles, PARTICLE_TOTAL * sizeof(Particle), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        printf("GPUassert: %s \n", cudaGetErrorString(error));
        */
    error = cudaMemcpy(VC, d_VC, 2 * PARTICLE_TOTAL * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        printf("GPUassert: %s \n", cudaGetErrorString(error));
    error = cudaMemcpy(colors, d_colors, 3 * PARTICLE_TOTAL * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        printf("GPUassert: %s \n", cudaGetErrorString(error));

    return cudaSuccess;
}
