# 🥚 Egg-CV System Walkthrough

> **A Guided Tour of the CNN-Based Automated Egg Grading System**

Welcome to the Egg-CV walkthrough! This guide will take you through the entire system, from setup to advanced features. Follow along step-by-step to learn how to use the system effectively.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Quick Start](#quick-start)
3. [Guided Tour](#guided-tour)
   - [Step 1: Create an Account](#step-1-create-an-account)
   - [Step 2: Log In](#step-2-log-in)
   - [Step 3: Upload Your First Image](#step-3-upload-your-first-image)
   - [Step 4: View Results](#step-4-view-results)
   - [Step 5: Explore the Dashboard](#step-5-explore-the-dashboard)
   - [Step 6: Check History](#step-6-check-history)
   - [Step 7: Adjust Settings](#step-7-adjust-settings)
4. [Key Workflows](#key-workflows)
5. [Tips & Best Practices](#tips--best-practices)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 System Overview

### What is Egg-CV?

Egg-CV is an intelligent egg grading system that uses **YOLOv8 object detection** and **CNN classification** to automatically:

| Feature | Description |
|---------|-------------|
| 🔍 **Detect Eggs** | Locates eggs in images or video frames |
| 🛡️ **Damage Detection** | Identifies damaged vs. intact eggs |
| 📏 **Size Grading** | Categorizes as Small, Medium, or Large |
| ⚖️ **Weight Estimation** | Estimates egg weight from size |
| 🏷️ **Grade Assignment** | Maps to standard grades (AA, A, B, Reject) |
| 📊 **Analytics** | Visual dashboard with statistics |

### How It Works

```
┌─────────────────┐
│   Upload Image  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLOv8 Detects│  ← Finds eggs with bounding boxes
│    Egg Regions  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Classify Each  │  ← Damaged or Not Damaged
│      Egg        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Grade & Size   │  ← Size + Damage = Grade
│   Calculation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Display Results │  ← Annotated image + data
└─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites Check

Before starting, ensure you have:

- [ ] **Python 3.8+** installed
- [ ] **Node.js 18+** installed
- [ ] **PostgreSQL 15+** running
- [ ] **YOLOv8 model file** at `models/egg_detection_finetuned/weights/best.pt`

### Step 1: Start the Backend

```bash
cd /Users/wii/Projects/python/egg-cv/backend

# Install dependencies
pip install -r requirements.txt

# Ensure .env is configured
# DATABASE_URL should point to your PostgreSQL instance
# SECRET_KEY should be set (min 32 characters)

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

✅ **Backend running at:** `http://localhost:8000`

### Step 2: Start the Frontend

```bash
cd /Users/wii/Projects/python/egg-cv/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

✅ **Frontend running at:** `http://localhost:5173`

### Step 3: Verify System Health

Visit: `http://localhost:8000/api/v1/dashboard/health`

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true
}
```

---

## 🎯 Guided Tour

Now let's walk through each feature of the system!

### Step 1: Create an Account

**🎯 Goal:** Register a new user account.

1. Open your browser and navigate to `http://localhost:5173`
2. You'll see the **Login Page** (default route)
3. Click **"Don't have an account? Register"** link
4. Fill in the registration form:

| Field | What to Enter |
|-------|----------------|
| Email | Your email address (e.g., `user@example.com`) |
| Username | Choose a username (e.g., `johndoe`) |
| Password | Secure password (min 8 characters) |

5. Click **"Register"**
6. ✅ **Success!** You're automatically logged in and redirected to the Dashboard

**📸 Screenshot placeholder:** `Registration page with form filled`

---

### Step 2: Log In

**🎯 Goal:** Learn the login process (for future visits).

1. If you're already logged in, click **Logout** in the top-right dropdown
2. You'll be redirected to `/login`
3. Enter your credentials:

| Field | Value |
|-------|-------|
| Email | The email you registered with |
| Password | Your password |

4. Click **"Login"**
5. ✅ **Success!** You're redirected to the Dashboard

**💡 Tip:** The system uses JWT tokens stored in localStorage for authentication. You'll stay logged in until you manually logout or the token expires (30 minutes).

---

### Step 3: Upload Your First Image

**🎯 Goal:** Upload an egg image for analysis.

1. From the Dashboard, click **"Upload"** in the navigation bar (or click the **Upload card**)
2. You'll see the **Upload Page** with a drag-and-drop zone
3. **Choose your upload method:**

   **Option A: Drag & Drop**
   - Drag an image file into the drop zone
   
   **Option B: Click to Browse**
   - Click the drop zone to open file picker
   - Select an image file (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`)

4. **Configure settings** (optional):

| Setting | Default | Description |
|---------|---------|-------------|
| Confidence Threshold | 0.75 | Minimum confidence for detection (0.1 - 0.99) |
| Save Annotated Image | ✅ Enabled | Save image with bounding boxes drawn |

5. Click **"Upload & Analyze"**
6. ✅ **Processing!** You'll see a loading spinner while the system analyzes the image
7. ✅ **Complete!** You'll be redirected to the **Results Page**

**📸 Screenshot placeholder:** `Upload page with file selected and settings visible`

**💡 Tips:**
- Use clear, well-lit images for best results
- Images with multiple eggs are supported
- The system processes both images and videos

---

### Step 4: View Results

**🎯 Goal:** Understand the prediction results.

After upload, you'll see the **Result Page** with:

#### 4.1 Annotated Image

- The uploaded image with **bounding boxes** drawn around each detected egg
- Each box is **color-coded** by grade:

| Grade | Color | Meaning |
|-------|-------|---------|
| AA | 🟢 Green | Large + Not Damaged |
| A | 🔵 Blue | Medium + Not Damaged |
| B | 🟡 Amber | Small + Not Damaged |
| N/A | ⚪ Gray | Size unknown |
| Reject | 🔴 Red | Damaged egg |

**📸 Screenshot placeholder:** `Result page showing annotated image with colored bounding boxes`

#### 4.2 Grade Summary

A card showing the breakdown:
- **Total Detections:** X eggs found
- **Not Damaged:** X eggs
- **Damaged:** X eggs

#### 4.3 Detection Details Table

| # | Grade | Size | Weight | Confidence | Actions |
|---|-------|------|--------|------------|---------|
| 1 | AA | Large | 65g | 0.92 | View Box |
| 2 | A | Medium | 55g | 0.88 | View Box |
| 3 | Reject | - | - | 0.95 | View Box |

#### 4.4 Download Annotated Image

- Click **"Download Annotated Image"** to save the processed image
- Great for documentation or sharing results

**💡 Tip:** Click on any detection row to highlight its bounding box on the image!

---

### Step 5: Explore the Dashboard

**🎯 Goal:** Understand your prediction history and statistics.

1. Click **"Dashboard"** in the navigation
2. You'll see the **Dashboard Page** with:

#### 5.1 Statistics Cards

| Card | Shows |
|------|-------|
| 📊 **Total Predictions** | Number of uploads you've made |
| 🖼️ **Images** | Number of image uploads |
| 🎥 **Videos** | Number of video uploads |
| 🥚 **Total Detections** | Total eggs detected across all uploads |

#### 5.2 Grade Distribution Chart

- A **pie chart** (Recharts) showing the breakdown of egg grades
- Hover over sections to see exact counts
- Colors match the grade colors from results

**📸 Screenshot placeholder:** `Dashboard with statistics cards and pie chart`

#### 5.3 Recent Predictions

- List of your last 5 uploads
- Shows: File name, type, date, detection count
- Click any item to view full results

#### 5.4 Quick Actions

- **Upload New Image** button → Jump to upload page
- **View All History** button → Jump to history page

**💡 Tip:** Use the dashboard to quickly assess your egg quality trends over time!

---

### Step 6: Check History

**🎯 Goal:** Browse and manage all your past predictions.

1. Click **"History"** in the navigation
2. You'll see the **History Page** with a table of all your uploads:

| File Name | Type | Date | Detections | Status | Actions |
|----------|------|------|------------|--------|---------|
| eggs_001.jpg | Image | 2026-05-04 | 5 | ✅ Completed | View / Delete |
| batch1.mp4 | Video | 2026-05-03 | 12 | ✅ Completed | View / Delete |
| test.png | Image | 2026-05-03 | 0 | ❌ Failed | Delete |

#### Features:

- **Filter by Status:** Use dropdown to show All / Completed / Failed
- **View Details:** Click "View" to see full results
- **Delete:** Click "Delete" to remove a prediction (with confirmation)
- **Pagination:** Automatically loads more as you scroll

**📸 Screenshot placeholder:** `History page with table and filters`

**💡 Tip:** Failed uploads can be deleted to keep your history clean!

---

### Step 7: Adjust Settings

**🎯 Goal:** Configure camera calibration for accurate size/weight measurements.

1. Click your **username** in the top-right
2. Select **"Settings"** from the dropdown menu
3. You'll see the **Settings Page**:

#### Camera Calibration

| Setting | Default | Description |
|---------|---------|-------------|
| mm_per_pixel | 0.5 | Calibration factor for size measurements |

**What is mm_per_pixel?**
- This value converts pixels to millimeters
- Used to calculate egg size (diameter) and estimate weight
- **Calibrate by:** Measuring a known object in the image

#### How to Calibrate:

1. Place a ruler or object of known size in the image
2. Measure its length in pixels (using image editor)
3. Calculate: `mm_per_pixel = actual_mm / pixels`
4. Enter the value and click **"Save Settings"**

**Example:**
- Ruler measures 100mm in real life
- In the image, it's 200 pixels long
- `mm_per_pixel = 100 / 200 = 0.5`

**📸 Screenshot placeholder:** `Settings page with calibration explanation`

**💡 Tip:** Recalibrate if you change the camera distance or lens zoom!

---

## 🔄 Key Workflows

### Workflow 1: Batch Processing

**Scenario:** You have multiple egg images to process.

```
1. Navigate to Upload page
2. Upload first image → Wait for results
3. Click "Upload" in nav → Upload second image
4. Repeat for all images
5. Visit History page to see all results
6. Use Dashboard to see overall statistics
```

### Workflow 2: Quality Control

**Scenario:** Checking a shipment of eggs for quality.

```
1. Upload image of egg batch
2. View results → Check grade distribution
3. Identify Reject (damaged) eggs
4. Download annotated image for documentation
5. Adjust camera calibration if needed (Settings)
6. Process more batches → Compare via Dashboard
```

### Workflow 3: Troubleshooting Poor Results

**Scenario:** Detections are inaccurate or missing.

```
1. Check image quality (lighting, focus)
2. Lower Confidence Threshold (Upload page)
3. Re-upload and compare results
4. Verify camera calibration (Settings → mm_per_pixel)
5. Check if model is loaded (Health endpoint)
6. Review failed uploads in History
```

---

## 💡 Tips & Best Practices

### Image Quality

| Do ✅ | Don't ❌ |
|-------|----------|
| Use well-lit images | Don't use dark/blurry images |
| Capture eggs clearly | Don't obscure eggs with objects |
| Use consistent distance | Don't vary zoom between shots |
| Calibrate camera properly | Don't guess mm_per_pixel |

### Efficient Workflow

1. **Calibrate once** - Set mm_per_pixel correctly before batch processing
2. **Use batch uploads** - Process multiple images in sequence
3. **Review History** - Keep track of all predictions in one place
4. **Monitor Dashboard** - Watch for trends in egg quality

### Security

- **Change SECRET_KEY** in production (min 32 characters)
- **Use strong passwords** for user accounts
- **Don't share JWT tokens** - they expire in 30 minutes
- **Logout** when done, especially on shared computers

---

## 🔧 Troubleshooting

### Issue: Can't Log In

**Symptoms:** "Invalid credentials" error

**Solutions:**
- [ ] Check email spelling
- [ ] Ensure Caps Lock is off
- [ ] Use "Register" if you haven't created an account
- [ ] Check if backend is running (`http://localhost:8000/docs`)

### Issue: Upload Fails

**Symptoms:** "Upload failed" or spinning indefinitely

**Solutions:**
- [ ] Check file type (must be .jpg, .png, .webp, .bmp, .mp4, .avi, .mov)
- [ ] Check file size (max 50MB)
- [ ] Verify backend is running
- [ ] Check browser console for errors (F12 → Console)

### Issue: No Eggs Detected

**Symptoms:** "Total detections: 0"

**Solutions:**
- [ ] Lower confidence threshold (try 0.5)
- [ ] Improve image lighting
- [ ] Ensure eggs are visible and not obscured
- [ ] Check if YOLO model is loaded (Health endpoint)

### Issue: Wrong Size/Weight

**Symptoms:** Eggs sized incorrectly

**Solutions:**
- [ ] Recalibrate camera (Settings → mm_per_pixel)
- [ ] Ensure calibration object is in same plane as eggs
- [ ] Verify measurement calculations
- [ ] Check if eggs are at consistent distance

### Issue: Database Connection Error

**Symptoms:** Backend won't start, "role does not exist"

**Solutions:**
- [ ] Verify PostgreSQL is running: `pg_isready`
- [ ] Check DATABASE_URL in `.env`
- [ ] Ensure database `eggcvdatabase` exists
- [ ] Verify user has proper permissions

### Issue: Frontend Can't Reach Backend

**Symptoms:** API requests fail, CORS errors

**Solutions:**
- [ ] Backend running on port 8000?
- [ ] Frontend proxy configured in `vite.config.ts`?
- [ ] Check browser Network tab (F12 → Network)
- [ ] Try accessing API directly: `http://localhost:8000/api/v1/dashboard/health`

---

## 📚 Additional Resources

| Resource | Location | Description |
|----------|----------|-------------|
| API Documentation | `http://localhost:8000/docs` | Interactive Swagger UI |
| Architecture Doc | `MVP_ARCHITECTURE.md` | Detailed system design |
| Original README | `README.md` | Project overview and setup |
| Progress Report | `PROGRESS_REPORT.md` | Development history |

---

## 🎉 Conclusion

Congratulations! You've completed the Egg-CV walkthrough. You now know how to:

- ✅ Set up and start the system
- ✅ Register and log in
- ✅ Upload and analyze egg images
- ✅ Interpret results and grades
- ✅ Use the dashboard for analytics
- ✅ Manage your prediction history
- ✅ Configure camera calibration
- ✅ Troubleshoot common issues

**Next Steps:**
1. Process your first batch of eggs
2. Experiment with different confidence thresholds
3. Calibrate your camera for accurate measurements
4. Explore the API documentation for advanced usage

**Need Help?**
- Check the troubleshooting section above
- Review `MVP_ARCHITECTURE.md` for technical details
- Inspect browser console (F12) for error messages

---

*Happy Egg Grading! 🥚✨*

---

**Document Version:** 1.0  
**Last Updated:** May 2026  
**System Version:** Egg-CV API 1.0.0
