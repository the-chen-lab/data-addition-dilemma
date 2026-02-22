# Radish

A React + Vite app that builds to a single static HTML file via `vite-plugin-singlefile`.

## Deploying

Run `./deploy.sh` from this directory. It will:
1. Build the app (`npm run build` → `dist/index.html`)
2. Copy `dist/index.html` to `../irenetrampoline.github.io/static/radish/index.html`
3. Run `make upload` in the Hugo site to publish to S3/CloudFront

## Local development

```
npm run dev
```
