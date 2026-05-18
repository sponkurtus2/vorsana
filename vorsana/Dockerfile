FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y \
    --default-toolchain 1.88.0 \
    --profile minimal

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src
COPY models ./models

RUN cargo build --release

FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/vorsana ./vorsana
COPY --from=builder /app/models ./models

ENV RUST_LOG=info

CMD ["./vorsana"]
