#pragma once

#include "../Definitions.hpp"
#include <iostream>
#include <cuda.h>
#include "../domain.hpp"
#include "../repository.hpp"
#include "../timer_cuda.hpp"

void launch_kernel(repository& repo, timer_cuda*);
void launch_kernel2(repository& repo, timer_cuda*, int);
void launch_kernel3(repository& repo, timer_cuda*);
extern int kernel_count;
