# GitHub Setup Instructions

## Step 1: Create a GitHub Repository

1. Go to https://github.com/new
2. Repository name: `edge_detection` (or any name you prefer)
3. Description: "Edge detection from scratch using NumPy - Sobel, Prewitt, and Canny algorithms"
4. Choose Public or Private
5. **DO NOT** check "Initialize with README" (we already have files)
6. Click "Create repository"

## Step 2: Copy Your Repository URL

After creating the repository, GitHub will show you a URL like:
- `https://github.com/YOUR_USERNAME/edge_detection.git`

Copy this URL.

## Step 3: Run These Commands

Open terminal in this directory and run:

```bash
# Add your GitHub repository as remote (replace with your actual URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin master
```

## Alternative: If you want to use main branch instead of master

```bash
# Rename branch to main
git branch -M main

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

## What's Already Done

✅ Git repository initialized
✅ All files committed
✅ Ready to push

You just need to:
1. Create the repository on GitHub
2. Add the remote URL
3. Push!



