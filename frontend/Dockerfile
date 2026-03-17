FROM node:20-slim
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
ARG API_URL=http://localhost:7860
ENV API_URL=$API_URL
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
