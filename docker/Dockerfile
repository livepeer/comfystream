ARG	BASE_IMAGE=livepeer/comfyui-base:latest

FROM	${BASE_IMAGE}

ENV	PATH="/workspace/miniconda3/bin:${PATH}" \
	NVM_DIR=/root/.nvm \
	NODE_VERSION=18.18.0

RUN	echo "Using base image: ${BASE_IMAGE}" && \
	apt update && \
	apt install -yqq supervisor && \
	wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

WORKDIR	/

# Install node and npm
RUN	bash -c "source $NVM_DIR/nvm.sh && \
	nvm install $NODE_VERSION  && \
	nvm alias default $NODE_VERSION && \
	nvm use default"

# Add node and npm to path so the commands are available
ENV	NODE_PATH="$NVM_DIR/v$NODE_VERSION/lib/node_modules" \
	PATH="$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH"

# Create the supervisor configuration file for ComfyUI and ComfyStream
COPY	docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY    --chmod=0755	docker/entrypoint.sh /workspace/comfystream/docker/entrypoint.sh

WORKDIR	/workspace/comfystream/ui
RUN npm install
RUN	npm run build

# Restore .gitkeep file after ui build for devcontainer users
RUN touch /workspace/comfystream/nodes/web/static/.gitkeep

WORKDIR	/workspace/comfystream

# Temp fix for deps
#RUN conda run -n comfystream --no-capture-output pip install "numpy<2.0"
#RUN conda run -n comfystream --no-capture-output pip install "opencv-contrib-python==4.9.0.80" "opencv-python-headless==4.9.0.80" "thinc==8.2.5"

EXPOSE	8188	8889	3000
EXPOSE	1024-65535/udp

ENTRYPOINT	[ "/workspace/comfystream/docker/entrypoint.sh" ]

CMD	[ "/bin/bash" ]
