# 🚀 Deployment Guide

This guide covers various deployment options for the CIFAR-10 Image Classification project.

## Table of Contents
- [Local Deployment](#local-deployment)
- [Hugging Face Spaces](#hugging-face-spaces)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)

---

## 🏠 Local Deployment

### Quick Start

```bash
# Activate your environment (if using conda/venv)
# conda activate your-env

# Navigate to project directory
cd "Project 4 - Image Classification"

# Launch the Gradio app
python app/gradio_app.py
```

The app will be available at:
- **Local**: http://127.0.0.1:7860
- **Public**: A temporary public URL will be generated (valid for 72 hours)

### Configuration Options

Edit `app/gradio_app.py` to customize:

```python
demo.launch(
    server_name="0.0.0.0",  # Listen on all interfaces
    server_port=7860,        # Port number
    share=True,              # Generate public URL
    theme=gr.themes.Soft()   # UI theme
)
```

---

## ☁️ Hugging Face Spaces

Deploy your model as a public web app on [Hugging Face Spaces](https://huggingface.co/spaces).

### Prerequisites
- Hugging Face account
- Git LFS installed (for model files)

### Step-by-Step Guide

#### 1. Create a New Space

```bash
# Visit https://huggingface.co/new-space
# - Name: cifar10-classifier (or your choice)
# - SDK: Gradio
# - Hardware: CPU Basic (free tier)
```

#### 2. Clone Your Space

```bash
git clone https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME
cd YOUR-SPACE-NAME
```

#### 3. Copy Project Files

```bash
# Copy necessary files
cp -r ../Project\ 4\ -\ Image\ Classification/app/* .
cp -r ../Project\ 4\ -\ Image\ Classification/src .
cp ../Project\ 4\ -\ Image\ Classification/requirements.txt .
cp ../Project\ 4\ -\ Image\ Classification/README.md .
```

#### 4. Upload Model Files

Model files (`.pth`) are large and require Git LFS:

```bash
# Initialize Git LFS
git lfs install

# Track model files
git lfs track "*.pth"
git add .gitattributes

# Copy your trained model
cp ../Project\ 4\ -\ Image\ Classification/models/best_model.pth models/
cp ../Project\ 4\ -\ Image\ Classification/models/model_metadata.json models/

# Copy outputs folder (if needed)
cp -r ../Project\ 4\ -\ Image\ Classification/outputs .
```

#### 5. Update README Frontmatter

Ensure your `README.md` starts with:

```yaml
---
title: CIFAR-10 Image Classifier
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: gradio_app.py
pinned: false
---
```

#### 6. Commit and Push

```bash
git add .
git commit -m "Initial deployment"
git push
```

Your Space will automatically build and deploy! 🎉

#### 7. Monitor Build

- Visit your Space URL: `https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME`
- Check the "Logs" tab if there are issues

### Updating Your Space

```bash
# Make changes locally
# Then push updates
git add .
git commit -m "Update model/app"
git push
```

---

## 🐳 Docker Deployment

Containerize your application for consistent deployment across environments.

### Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY outputs/ ./outputs/

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "app/gradio_app.py"]
```

### Build and Run

```bash
# Build the Docker image
docker build -t cifar10-classifier .

# Run the container
docker run -p 7860:7860 cifar10-classifier

# With GPU support (requires nvidia-docker)
docker run --gpus all -p 7860:7860 cifar10-classifier
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - GRADIO_SERVER_NAME=0.0.0.0
```

Run with:
```bash
docker-compose up
```

---

## ☁️ Cloud Deployment

### AWS (Amazon Web Services)

#### Using EC2

1. **Launch an EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: `g4dn.xlarge` (with GPU) or `t3.medium` (CPU only)
   - Security Group: Open port 7860

2. **Connect and Setup**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip git

# Clone your repository
git clone <your-repo-url>
cd "Project 4 - Image Classification"

# Install Python packages
pip3 install -r requirements.txt

# Run the app
python3 app/gradio_app.py
```

3. **Keep Running (Optional)**
```bash
# Use tmux or screen
sudo apt install tmux
tmux new -s gradio
python3 app/gradio_app.py
# Press Ctrl+B, then D to detach
```

#### Using AWS Lambda + API Gateway

For serverless deployment, convert to a REST API:
- Package model and code
- Create Lambda function (Python 3.10)
- Set up API Gateway trigger
- Note: Lambda has 10GB memory limit

### Google Cloud Platform (GCP)

#### Using Cloud Run

1. **Build and Push Container**
```bash
# Build for Cloud Run
docker build -t gcr.io/YOUR-PROJECT-ID/cifar10-classifier .

# Push to Google Container Registry
docker push gcr.io/YOUR-PROJECT-ID/cifar10-classifier
```

2. **Deploy**
```bash
gcloud run deploy cifar10-classifier \
  --image gcr.io/YOUR-PROJECT-ID/cifar10-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Microsoft Azure

#### Using Azure Container Instances

```bash
# Create resource group
az group create --name cifar10-rg --location eastus

# Deploy container
az container create \
  --resource-group cifar10-rg \
  --name cifar10-app \
  --image YOUR-DOCKER-IMAGE \
  --dns-name-label cifar10-classifier \
  --ports 7860
```

---

## 🔒 Security Considerations

### For Production Deployments

1. **Authentication**
   - Add user authentication to Gradio interface
   - Use environment variables for secrets
   - Implement rate limiting

2. **HTTPS**
   - Use reverse proxy (nginx) with SSL certificate
   - Let's Encrypt for free SSL certificates

3. **Monitoring**
   - Set up logging and monitoring
   - Track API usage and performance
   - Alert on errors or high load

### Example nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 🧪 Testing Your Deployment

### Health Check Endpoint

Add to your `gradio_app.py`:

```python
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://your-deployment-url/

# Using locust (Python)
pip install locust
locust -f load_test.py --host=http://your-deployment-url
```

---

## 📊 Performance Optimization

### For Production

1. **Model Optimization**
   - Use TorchScript for faster inference
   - Consider model quantization (INT8)
   - Batch predictions when possible

2. **Caching**
   - Cache model in memory
   - Use Redis for result caching

3. **Scaling**
   - Horizontal scaling with load balancer
   - Auto-scaling based on traffic
   - GPU instances for high throughput

---

## 📚 Additional Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [Hugging Face Spaces Guide](https://huggingface.co/docs/hub/spaces)
- [Docker Documentation](https://docs.docker.com/)
- [AWS EC2 Guide](https://docs.aws.amazon.com/ec2/)
- [GCP Cloud Run](https://cloud.google.com/run/docs)

---

## 🆘 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find and kill process on port 7860
lsof -ti:7860 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :7860   # Windows
```

**Out of Memory**
```python
# Reduce batch size in inference
# Use mixed precision (FP16)
# Close unnecessary applications
```

**Model Not Found**
```bash
# Verify model path
ls models/best_model.pth

# Check model_metadata.json exists
cat models/model_metadata.json
```

---

**Need help?** Open an issue on GitHub or check the [CONTRIBUTING.md](../CONTRIBUTING.md) guide.