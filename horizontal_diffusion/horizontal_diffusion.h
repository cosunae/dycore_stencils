#pragma once

#include "../Definitions.hpp"
#include <iostream>
#include <cuda.h>
#include "../domain.hpp"
#include "../repository.hpp"
#include "../timer_cuda.hpp"

void launch_kernel(repository& repo, timer_cuda*);
