# TBSentinel Presentation Manual QA Checklist

**Run this checklist immediately before your presentation to ensure 100% confidence.**

---

## Prerequisites
- ✅ Docker containers running: `docker compose up -d`
- ✅ Browser open at: http://localhost:8501
- ✅ Have a chest X-ray image ready (JPG/PNG)

---

## Step-by-Step Manual Test (5-7 Steps)

### **Step 1: Login**
1. Navigate to http://localhost:8501
2. Verify login page loads without errors
3. Enter username: `admin`
4. Enter password: `tbsentinel2026`
5. Click **Login** button
6. ✅ **PASS IF:** Dashboard loads successfully and shows "System ready" status

---

### **Step 2: Verify System Status**
1. Look at the sidebar (left panel)
2. Verify green pill shows "System ready" or "Model loaded and ready"
3. Verify model name is displayed (e.g., "multiclass_efficientnet_b3")
4. ✅ **PASS IF:** All status indicators are green and model is loaded

---

### **Step 3: Navigate to New Assessment**
1. Click **New Assessment** in the sidebar navigation
2. Verify the assessment page loads without errors
3. Verify patient information form is visible
4. Verify upload box shows "No image selected" placeholder
5. ✅ **PASS IF:** Page loads cleanly, no errors, form and upload box visible

---

### **Step 4: Run Complete Assessment**
1. Enter patient details (optional but recommended):
   - Patient name: `Test Patient`
   - Age: `35`
   - Gender: `Male`
   - Click **🪄 Generate** to auto-generate Patient ID
2. Click **Choose a chest X-ray image…** and select your test image
3. Verify image preview appears
4. Click **Run assessment** button (primary button)
5. Wait for progress bar to complete (10-30 seconds)
6. ✅ **PASS IF:**
   - Progress completes without errors
   - Prediction result appears with large confidence percentage
   - Grad-CAM images load (Original, Heatmap, Overlay side-by-side)
   - Clinical interpretation appears

---

### **Step 5: Verify Results Display**
1. Check the **Diagnostic result** card:
   - Verify color-coded badge (green/red/amber)
   - Verify large confidence percentage (e.g., "92%")
   - Click **"View all class probabilities"** expander and verify it opens
2. Check the **Grad-CAM attention map** card:
   - Verify 3 images display cleanly (no bounding boxes or debug text)
   - Verify "Original radiograph", "Activation map", "Overlay" captions
3. Check the **Clinical interpretation** card:
   - Verify short, punchy bullet points (3 items)
   - Verify no dense technical jargon is visible by default
4. ✅ **PASS IF:** All cards display correctly, UI is clean and simplified

---

### **Step 6: Save Assessment**
1. Scroll to **"💾 Save assessment"** card
2. Verify button is enabled (not grayed out)
3. Click **"💾 Save assessment to patient history"** button
4. Verify success toast appears: "Assessment [ID] saved"
5. Verify button changes to "✅ Saved as [ID]"
6. ✅ **PASS IF:** Save completes and success message appears

---

### **Step 7: View Patient History**
1. Click **Patient History** in sidebar navigation
2. Verify history page loads
3. Verify your saved assessment appears in the list
4. Click on the assessment record
5. Verify detail view opens with same results as Step 5
6. ✅ **PASS IF:** Assessment is saved and can be reopened with all data intact

---

## Final Verification

### **Optional: Export Test**
1. In the saved assessment detail view, click **"⬇ Download summary (JSON)"**
2. Verify file downloads successfully
3. ✅ **PASS IF:** JSON file downloads without errors

---

## Quick Decision Tree

| Step | Pass Criteria | Action if Fail |
|------|---------------|----------------|
| 1. Login | Dashboard loads | Check Docker containers: `docker compose ps` |
| 2. Status | Green "System ready" | Check API: `docker compose logs api` |
| 3. Navigation | Page loads, no errors | Refresh browser, check console (F12) |
| 4. Assessment | Prediction + Grad-CAM appear | Check API health, verify image format |
| 5. Results | Clean UI, no technical jargon | UI simplification already applied |
| 6. Save | Success toast appears | Check database permissions |
| 7. History | Assessment visible | Check SQLite database access |

---

## Emergency Fixes

**If Step 1 fails (login error):**
```bash
docker compose restart dashboard
```

**If Step 2 fails (API offline):**
```bash
docker compose logs api
docker compose restart api
```

**If Step 4 fails (prediction error):**
- Verify image is JPG/PNG/BMP
- Try a different test image
- Check API logs: `docker compose logs -f api`

**If Step 6 fails (save error):**
```bash
docker compose exec dashboard ls -la /app/data/patient_records
```

---

## ✅ All Steps Passed?

If all 7 steps pass, you are **100% ready for your presentation**.

**Good luck with your presentation! 🎓**
