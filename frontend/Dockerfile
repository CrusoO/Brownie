FROM node:24-alpine AS deps

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:24-alpine AS builder

ARG NEXT_PUBLIC_BACKEND_WS_URL=ws://localhost:8000/ws/chat
ARG NEXT_PUBLIC_BACKEND_HTTP_URL=http://localhost:8000

ENV NEXT_TELEMETRY_DISABLED=1
ENV NEXT_PUBLIC_BACKEND_WS_URL=${NEXT_PUBLIC_BACKEND_WS_URL}
ENV NEXT_PUBLIC_BACKEND_HTTP_URL=${NEXT_PUBLIC_BACKEND_HTTP_URL}

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:24-alpine AS runner

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000

WORKDIR /app
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/package-lock.json ./package-lock.json
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules

EXPOSE 3000

CMD ["npm", "run", "start"]
