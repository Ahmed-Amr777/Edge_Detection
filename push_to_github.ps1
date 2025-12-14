# PowerShell script to push to GitHub
# Make sure you've created the repository on GitHub first!

param(
    [Parameter(Mandatory=$true)]
    [string]$RepoUrl
)

Write-Host "Adding remote repository..." -ForegroundColor Green
git remote add origin $RepoUrl

Write-Host "Pushing to GitHub..." -ForegroundColor Green
git push -u origin master

Write-Host "`nDone! Your code is now on GitHub!" -ForegroundColor Green
Write-Host "Repository URL: $RepoUrl" -ForegroundColor Cyan



