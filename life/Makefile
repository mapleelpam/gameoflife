
# This is a GNU Makefile.

# It can be used to compile an OpenCL program with
# the Altera Beta OpenCL Development Kit.
# See README.txt for more information.


# You must configure ALTERAOCLSDKROOT to point the root directory of the Altera SDK for OpenCL
# software installation.
# See doc/getting_started.txt for more information on installing and
# configuring the Altera SDK for OpenCL.


# Creating a static library
TARGET = gameoflife_host 

# Where is the Altera SDK for OpenCL software?
#ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
#$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation)
#endif
#ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
#$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation.)
#endif

# Libraries to use, objects to compile
OBJ_DIR = obj
OBJ_EXT = o
SRCS = main.shared_mem.cpp 


# OpenCL compile and link flags.
#AOCL_COMPILE_CONFIG=$(shell aocl compile-config)
AOCL_COMPILE_CONFIG=-I/root/opencl/include
#AOCL_LINK_CONFIG=$(shell aocl link-config)
AOCL_LINK_CONFIG=-L/root/opencl/lib/ -lalteracl /root/opencl/lib/libalterahalmmd.so /root/opencl/lib/libalterammdpcie.so -lelf -lrt
AOCL_COMPILER?=g++

# Make it all!
#all : Makefile.linux

all :  gfx.o
	$(AOCL_COMPILER) $(SRCS) $^ -o $(TARGET) -DLINUX $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) -I/usr/X11R6/include -L/usr/X11R6/lib -lX11 -lm  -g

# Standard make targets
clean :
	@rm -f $(OBJS) $(TARGET)
