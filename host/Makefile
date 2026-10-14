# Desktop test build: make -C host
# Cross-check on big-endian PowerPC: make -C host ppc  (needs powerpc-linux-gnu-gcc + qemu-user)
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Wno-unused-parameter

all: iiama2-host

iiama2-host: host_main.c ../source/llm.c ../source/llm.h
	$(CC) $(CFLAGS) -o $@ host_main.c ../source/llm.c -lm

# Same code, compiled for 32-bit big-endian PowerPC like the Wii's Broadway.
# Run with: qemu-ppc ./iiama2-host-ppc model.bin tokenizer.bin ...
ppc: iiama2-host-ppc
iiama2-host-ppc: host_main.c ../source/llm.c ../source/llm.h
	powerpc-linux-gnu-gcc $(CFLAGS) -mcpu=750 -static -o $@ host_main.c ../source/llm.c -lm

clean:
	rm -f iiama2-host iiama2-host-ppc

.PHONY: all ppc clean
