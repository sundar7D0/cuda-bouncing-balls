void PostprocessCUDA(cudaGraphicsResource_t& dst, cudaGraphicsResource_t& src,
					unsigned int width, unsigned int height,
					float* filter,	//filter is assumed to be a 5x5 filter kernel
					float scale, float offset );