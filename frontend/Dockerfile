# Stage 1: build the React app.
FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY index.html tsconfig.json vite.config.ts ./
COPY src/ src/
RUN npm run build

# Stage 2: nginx serves the build, falls back to index.html for client-side routes and proxies
# /api to the backend (nginx.conf; SSE buffering is off for POST /api/chat).
FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/ /usr/share/nginx/html/
EXPOSE 8080
