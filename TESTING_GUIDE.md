# Testing Guide - AI Video Analysis App

## ✅ All Critical Fixes Applied

### Bugs Fixed:
1. **✅ Stale Job Timeout** - Increased from 20 min → 120 min
2. **✅ Transcription Timeout** - Wrapped with 8-minute timeout + heartbeat monitoring  
3. **✅ Progress Bar Overflow** - Capped at 100% in both SQL code paths
4. **✅ API Retry Logic** - 3 attempts with exponential backoff for OpenAI 500 errors

---

## 🧪 How to Test (Via Streamlit UI)

### Step 1: Open the App
The app is already running at the URL shown in your Replit webview.

### Step 2: Upload Test Video
1. Click the **file uploader** in the sidebar
2. Navigate to: `test videos/The Patient Encounter.mp4`
3. Upload the file (120 MB, 11 minutes)

### Step 3: Configure Settings (Optional)
- **Profile**: Medical Assessment (default)
- **FPS**: 1.0 (default - good balance)
- **Batch size**: 5 (default)
- **Resolution**: 720p (default)

### Step 4: Start Analysis
Click the **"Start Analysis"** button

### Step 5: Monitor Progress
- Click **"Refresh Status"** button periodically
- You should see progress through these stages:
  1. ✅ Extracting audio (~30 sec)
  2. ✅ Transcribing audio (~3-6 min)
  3. ✅ Analyzing frames (~8-12 min) - Progress bar will update
  4. ✅ Creating narrative (~1-2 min)
  5. ✅ Generating assessment (~30 sec)

**Expected Total Time:** 12-20 minutes

### Step 6: Download Results
Once complete, you'll see 3 download buttons:
- **📄 Download PDF Report ☁️** - From cloud storage
- **📝 Download Transcript** - Audio transcription
- **📖 Download Narrative** - Clinical observations

---

## 🔍 Monitoring from Terminal (Optional)

While the job runs via UI, you can monitor progress in terminal:

```bash
python3 monitor_job.py
```

This will show real-time updates of the most recent job.

---

## 🐛 What to Watch For

### ✅ Good Signs:
- Progress bar increases steadily
- No "stale job" timeout (jobs can run for 2 hours now)
- Progress stays at ≤100%
- If OpenAI API errors occur, automatic retry with backoff

### ⚠️ Potential Issues:
- **OpenAI API 500 errors** - Will auto-retry up to 3 times
- **Rate limits** - Will auto-retry with exponential backoff
- **Network timeouts** - Will retry automatically

---

## 📊 Expected Processing Timeline

For 11-minute video (The Patient Encounter.mp4):

| Stage | Time | Notes |
|-------|------|-------|
| Audio Extraction | ~30 sec | FFmpeg extraction |
| Audio Transcription | 3-6 min | Whisper API with timeout protection |
| Frame Analysis | 8-12 min | Main processing stage with progress bar |
| Narrative Synthesis | 1-2 min | Combining observations |
| Assessment Generation | ~30 sec | PDF creation |
| **Total** | **12-20 min** | Varies with API response times |

---

## 🔄 If Test Fails

### Common Failure Scenarios:

1. **OpenAI API 500 Error (Server Error)**
   - **What it means:** OpenAI's servers are temporarily unavailable
   - **What happens:** Auto-retries 3 times with backoff (2s, 4s, 8s)
   - **Action:** Wait 10 minutes and try again

2. **Progress Stuck**
   - **What it means:** API call might be slow
   - **What happens:** Heartbeat updates every 30 seconds prove job is alive
   - **Action:** Wait - jobs can run for up to 2 hours now

3. **Transcription Timeout**
   - **What it means:** Whisper API took >8 minutes
   - **What happens:** Job fails gracefully with error message
   - **Action:** Retry - usually succeeds on second attempt

### How to Retry:
Simply upload the video again and click "Start Analysis"

---

## ✨ Success Criteria

Test is successful when:
- ✅ Job completes to 100%
- ✅ Status shows "COMPLETED"
- ✅ All 3 download buttons work
- ✅ PDF report generated and uploaded to cloud storage
- ✅ Transcript and narrative are complete

---

## 📝 Current Test Status

**Test Video:** `test videos/The Patient Encounter.mp4`
- Duration: 11 minutes 13 seconds
- Size: 120 MB
- Resolution: 1280x720 (720p HD)

**App Status:** ✅ Running and ready for testing

**Fixes Applied:** ✅ All 4 critical bugs fixed

**Ready to Test:** ✅ Yes - use Streamlit UI upload method

---

## 💡 Tips for Successful Testing

1. **Use the UI** - Don't test via scripts (threading issues)
2. **Be patient** - 11-min video takes 12-20 min to process
3. **Monitor progress** - Use "Refresh Status" button
4. **Check heartbeats** - Updates every 30 seconds prove job is alive
5. **Retry if needed** - OpenAI API can have transient errors

---

## 🎯 Next Steps After Successful Test

Once the test completes successfully:
1. ✅ Verify all outputs (PDF, Transcript, Narrative)
2. ✅ Test with different videos/lengths
3. ✅ Consider deploying to Reserved VM for production use
4. ✅ Monitor performance metrics (processing time, API costs)

---

**Happy Testing! 🚀**
