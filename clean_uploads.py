import os
import glob

def clean_uploads():
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    if not os.path.exists(uploads_dir):
        print(f"Uploads directory does not exist: {uploads_dir}")
        return
    files = glob.glob(os.path.join(uploads_dir, '*'))
    deleted = 0
    for f in files:
        if os.path.isfile(f) and not f.endswith('.gitkeep'):
            try:
                os.remove(f)
                deleted += 1
            except Exception as e:
                print(f"Could not delete {f}: {e}")
    print(f"Cleaned {deleted} files from uploads folder.")

if __name__ == "__main__":
    clean_uploads()
