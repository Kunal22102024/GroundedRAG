# Phase 1: nginx serving a static status page and proxying /api.
# Phase 5 turns this into a multi-stage build (node build -> nginx) for the
# React app; nginx.conf stays as it is.
FROM nginxinc/nginx-unprivileged:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY static/ /usr/share/nginx/html/

EXPOSE 8080
