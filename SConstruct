import os
import nvcc

env = Environment()

env['CUDA_TOOLKIT_PATH'] = '/usr/lib/cuda'

env.Tool('nvcc', toolpath = ['./'])
env['CXXFLAGS'] = ['-O3', '-g']
env['NVCCFLAGS'] = ['-dc']

sources = Glob("**/**/*.cpp") + \
          Glob("**/**/**/*.cpp") + \
          Glob("**/**/*.cu") + \
          Glob("**/**/**/*.cu")

#for file in Glob("**/*.cu"):
#    print(file)

#print([x.rfile() for x in sources])

env.Program('backpropagate', sources + Glob("src/main.cu"))
env.Program('solve_problem', sources + Glob("src/solve_problem.cu"))
env.Program('benchmark_cpu', sources + Glob("src/benchmark_cpu.cu"))
env.Program('benchmark_gpu', sources + Glob("src/benchmark_gpu.cu"))
