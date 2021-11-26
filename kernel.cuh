#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Particle.h"
#include "utils.h"

#define PARTICLE_BLOCK_WIDTH 1000
#define PARTICLE_BLOCK_HEIGHT 1000
#define PARTICLE_TOTAL PARTICLE_BLOCK_WIDTH*PARTICLE_BLOCK_HEIGHT

#define THREAD_BLOCK_SIZE 1024
#define BLOCK_COUNT PARTICLE_TOTAL/THREAD_BLOCK_SIZE+1

typedef struct {
	bool LMB;
	sf::Vector2f mousePos;
	float dt;
} param;

cudaError_t computeParticles(Particle* particles, Particle* d_particles, param* p, float* VC, float* d_VC,
	unsigned char* colors, unsigned char* d_colors);
__global__ void computeColors(Particle* p, param* par, unsigned char* colors);
__global__ void computeForces(Particle* p, param* par, float* VC);
__global__ void computePositions(Particle* p, param* par, float* VC);
