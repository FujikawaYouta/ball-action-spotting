NAME?=ball-action-spotting
COMMAND?=bash
OPTIONS?=

GPUS?=device=0  # gpu 0
ifeq ($(GPUS),none)
	GPUS_OPTION=
else
	GPUS_OPTION=--gpus $(GPUS)
endif

.PHONY: all
all: stop build run

.PHONY: build
build:
	docker build -t ball-container .

.PHONY: stop
stop:
	-docker stop ball-container
	-docker rm ball-container

.PHONY: run
run:
	docker run --rm -dit \
		--net=host \
		--ipc=host \
		$(OPTIONS) \
		$(GPUS_OPTION) \
		-v $(shell pwd):/workdir \
		--name=ball-container \
		ball-container \
		$(COMMAND)
	docker attach ball-container

.PHONY: attach
attach:
	docker attach ball-container

.PHONY: logs
logs:
	docker logs -f ball-container

.PHONY: exec
exec:
	docker exec -it $(OPTIONS) ball-container $(COMMAND)
