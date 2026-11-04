# One image: builds the React front end, then serves it with the Python API.
#   docker build -t epoxy-explorer .
#   docker run --rm -p 8000:8000 epoxy-explorer   ->  http://localhost:8000
ARG NODE_IMAGE=node:20-alpine
# Full (non-slim) Python image: it ships libgomp, which XGBoost needs.
ARG PY_IMAGE=python:3.11

FROM ${NODE_IMAGE} AS web
WORKDIR /web
COPY app/web/package.json app/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY app/web/index.html app/web/vite.config.js ./
COPY app/web/src ./src
RUN npm run build

FROM ${PY_IMAGE}
WORKDIR /repo
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
COPY requirements.txt app/requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt
COPY data ./data
COPY src ./src
COPY results ./results
COPY app/api ./app/api
COPY --from=web /web/dist ./app/web/dist
WORKDIR /repo/app/api
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
