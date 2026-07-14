FROM node:18 as build

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .

ARG REACT_APP_API_URL=http://48.192.178.39:8000
ENV REACT_APP_API_URL=$REACT_APP_API_URL

RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
