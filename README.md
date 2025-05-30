# Bone Fracture Detection Web Application

This is a simple web application that uses a pre-trained PyTorch model to detect bone fractures in X-ray images.

## Features

- Upload X-ray images
- Automatic prediction of bone fractures
- Display of prediction results (Positive/Negative)

## Local Setup and Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Deploying to Vercel

This application is configured to be deployed on Vercel. Follow these steps to deploy:

### Option 1: Deploy with Vercel CLI

1. Install the Vercel CLI:
   ```
   npm install -g vercel
   ```

2. Login to Vercel:
   ```
   vercel login
   ```

3. Deploy the application:
   ```
   vercel
   ```

### Option 2: Deploy with GitHub

1. Create a GitHub repository and push your code:
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. Go to [Vercel](https://vercel.com) and sign in with your GitHub account.

3. Click "New Project" and import your GitHub repository.

4. Configure the project settings:
   - Framework Preset: Other
   - Root Directory: ./
   - Build Command: None
   - Output Directory: None

5. Click "Deploy" and wait for the deployment to complete.

## Usage

1. Click the "Choose File" button to select an X-ray image
2. Click "Detect Fracture" to upload the image and get a prediction
3. View the result (Positive = Fracture detected, Negative = No fracture detected)

## Technical Details

- The application uses a CNN model trained to detect bone fractures
- Images are preprocessed to match the model's expected input format
- The model outputs a prediction value between 0 and 1, with values ≥ 0.5 indicating a positive result (fracture detected)
- For the Vercel deployment, a simplified version is used that generates random predictions

## Troubleshooting

If you encounter any issues:

1. Check that you have all the required dependencies installed
2. Ensure the uploaded images are in a supported format (JPG, JPEG, PNG)
3. For Vercel deployment issues, check the Vercel logs for more information
