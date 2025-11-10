import json
import os
import shutil
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

class BatchManager:
    """
    Persistent batch management system that stores results on disk,
    not in session state. Survives app resets gracefully.
    """
    
    def __init__(self, base_dir: str = "batches"):
        self.base_dir = base_dir
        self.index_file = os.path.join(base_dir, "index.json")
        self.latest_batch_file = os.path.join(base_dir, "latest_batch_id.txt")
        
        # Ensure directories exist
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs("html_reports", exist_ok=True)
        os.makedirs("narratives", exist_ok=True)
    
    def create_batch(self, video_names: List[str], settings: Dict[str, Any]) -> str:
        """Create a new batch and return batch_id"""
        # Generate unique batch ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = str(uuid.uuid4())[:8]
        batch_id = f"batch_{timestamp}_{short_id}"
        
        # Create batch directory
        batch_dir = os.path.join(self.base_dir, batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        
        # Create initial manifest
        manifest = {
            "batch_id": batch_id,
            "created_at": datetime.now().isoformat(),
            "settings": settings,
            "status": "running",
            "items": []
        }
        
        # Initialize items for each video
        for video_name in video_names:
            manifest["items"].append({
                "input_name": video_name,
                "detected_pattern": settings.get("pattern", "unknown"),
                "status": "pending",
                "html_path": None,
                "narrative_path": None,
                "score": None,
                "started_at": None,
                "completed_at": None,
                "error": None
            })
        
        # Write manifest
        self._write_manifest(batch_id, manifest)
        
        # Update index
        self._update_index(batch_id, len(video_names))
        
        # Mark as latest batch
        with open(self.latest_batch_file, "w") as f:
            f.write(batch_id)
        
        return batch_id
    
    def update_item_status(self, batch_id: str, video_name: str, status: str, 
                          html_path: Optional[str] = None, narrative_path: Optional[str] = None,
                          score: Optional[str] = None, error: Optional[str] = None):
        """Update status of a specific video in the batch"""
        manifest = self._load_manifest(batch_id)
        if not manifest:
            return False
        
        # Find and update the item
        for item in manifest["items"]:
            if item["input_name"] == video_name:
                item["status"] = status
                if status == "processing":
                    item["started_at"] = datetime.now().isoformat()
                elif status in ["completed", "failed"]:
                    item["completed_at"] = datetime.now().isoformat()
                
                if html_path:
                    item["html_path"] = html_path
                if narrative_path:
                    item["narrative_path"] = narrative_path
                if score:
                    item["score"] = score
                if error:
                    item["error"] = error
                break
        
        # Check if all items are complete
        all_completed = all(item["status"] in ["completed", "failed"] 
                           for item in manifest["items"])
        if all_completed:
            manifest["status"] = "completed"
            manifest["finished_at"] = datetime.now().isoformat()
            
            # Update index status to completed
            self._update_batch_status_in_index(batch_id, "completed")
        
        # Write updated manifest
        self._write_manifest(batch_id, manifest)
        return True
    
    def get_batch_manifest(self, batch_id: str) -> Optional[Dict]:
        """Load batch manifest from disk"""
        return self._load_manifest(batch_id)
    
    def list_batches(self) -> List[Dict]:
        """List all batches from index"""
        if not os.path.exists(self.index_file):
            return []
        
        try:
            with open(self.index_file, "r") as f:
                index = json.load(f)
            return sorted(index.get("batches", []), 
                         key=lambda x: x["created_at"], reverse=True)
        except:
            return []
    
    def get_latest_batch_id(self) -> Optional[str]:
        """Get the most recent batch ID"""
        if not os.path.exists(self.latest_batch_file):
            return None
        
        try:
            with open(self.latest_batch_file, "r") as f:
                return f.read().strip()
        except:
            return None
    
    def create_batch_zip(self, batch_id: str) -> Optional[str]:
        """Create a ZIP file containing all batch results"""
        manifest = self._load_manifest(batch_id)
        if not manifest:
            return None
        
        batch_dir = os.path.join(self.base_dir, batch_id)
        zip_path = os.path.join(batch_dir, f"{batch_id}.zip")
        
        # Skip if ZIP already exists and is newer than manifest
        manifest_path = os.path.join(batch_dir, "manifest.json")
        if (os.path.exists(zip_path) and 
            os.path.getmtime(zip_path) > os.path.getmtime(manifest_path)):
            return zip_path
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add manifest
                zf.write(manifest_path, "manifest.json")
                
                # Add all completed files
                for item in manifest["items"]:
                    if item["status"] == "completed":
                        # Add HTML file
                        if item["html_path"] and os.path.exists(item["html_path"]):
                            zf.write(item["html_path"], 
                                   f"html_reports/{os.path.basename(item['html_path'])}")
                        
                        # Add narrative file
                        if item["narrative_path"] and os.path.exists(item["narrative_path"]):
                            zf.write(item["narrative_path"], 
                                   f"narratives/{os.path.basename(item['narrative_path'])}")
                
                # Create and add summary CSV
                summary_csv = self._create_summary_csv(manifest)
                zf.writestr("batch_summary.csv", summary_csv)
            
            return zip_path
        except Exception as e:
            print(f"Error creating ZIP: {e}")
            return None
    
    def _write_manifest(self, batch_id: str, manifest: Dict):
        """Atomically write manifest to disk"""
        batch_dir = os.path.join(self.base_dir, batch_id)
        manifest_path = os.path.join(batch_dir, "manifest.json")
        temp_path = manifest_path + ".tmp"
        
        # Write to temp file first
        with open(temp_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Atomic rename
        os.rename(temp_path, manifest_path)
    
    def _load_manifest(self, batch_id: str) -> Optional[Dict]:
        """Load manifest from disk"""
        manifest_path = os.path.join(self.base_dir, batch_id, "manifest.json")
        
        if not os.path.exists(manifest_path):
            return None
        
        try:
            with open(manifest_path, "r") as f:
                return json.load(f)
        except:
            return None
    
    def _update_index(self, batch_id: str, video_count: int):
        """Update the batch index"""
        # Load existing index
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r") as f:
                    index = json.load(f)
            except:
                index = {"batches": []}
        else:
            index = {"batches": []}
        
        # Add this batch
        batch_entry = {
            "batch_id": batch_id,
            "created_at": datetime.now().isoformat(),
            "video_count": video_count,
            "status": "running"
        }
        
        # Remove any existing entry (in case of overwrites)
        index["batches"] = [b for b in index["batches"] if b["batch_id"] != batch_id]
        index["batches"].append(batch_entry)
        
        # Keep only last 50 batches
        if len(index["batches"]) > 50:
            index["batches"] = index["batches"][-50:]
        
        # Write atomically
        temp_path = self.index_file + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(index, f, indent=2)
        os.rename(temp_path, self.index_file)
    
    def _update_batch_status_in_index(self, batch_id: str, status: str):
        """Update batch status in index.json"""
        if not os.path.exists(self.index_file):
            return
        
        try:
            with open(self.index_file, "r") as f:
                index = json.load(f)
        except:
            return
        
        # Find and update the batch entry
        for batch_entry in index.get("batches", []):
            if batch_entry["batch_id"] == batch_id:
                batch_entry["status"] = status
                if status == "completed":
                    batch_entry["finished_at"] = datetime.now().isoformat()
                break
        
        # Write atomically
        temp_path = self.index_file + ".tmp" 
        with open(temp_path, "w") as f:
            json.dump(index, f, indent=2)
        os.rename(temp_path, self.index_file)
    
    def _create_summary_csv(self, manifest: Dict) -> str:
        """Create a CSV summary of batch results"""
        lines = ["Video Name,Pattern,Status,Score,HTML File,TXT File"]
        
        for item in manifest["items"]:
            html_filename = os.path.basename(item["html_path"]) if item["html_path"] else ""
            txt_filename = os.path.basename(item["narrative_path"]) if item["narrative_path"] else ""
            
            lines.append(f'"{item["input_name"]}","{item["detected_pattern"]}",'
                        f'"{item["status"]}","{item.get("score", "")}","'
                        f'{html_filename}","{txt_filename}"')
        
        return "\n".join(lines)