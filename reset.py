import os
import shutil

# --- CONFIGURATION ---
DB_PATH = "./my_offline_db"
DATA_PATH = "./my_data"

def main():
    print("⚠️  WARNING: This will wipe the AI's memory.")
    print(f"Target Database: {os.path.abspath(DB_PATH)}")
    
    confirm = input("\nAre you sure you want to delete the database? (yes/no): ")
    
    if confirm.lower() == "yes":
        # 1. Delete the Vector Database
        if os.path.exists(DB_PATH):
            try:
                shutil.rmtree(DB_PATH)
                print(f"✅ Deleted database folder: {DB_PATH}")
                print("   (The AI is now a blank slate)")
            except Exception as e:
                print(f"❌ Error deleting database: {e}")
        else:
            print("ℹ️  No database found to delete.")

        # 2. Optional: Clear the source files too?
        clear_files = input("\nDo you also want to delete all files inside 'my_data'? (yes/no): ")
        if clear_files.lower() == "yes":
            if os.path.exists(DATA_PATH):
                for filename in os.listdir(DATA_PATH):
                    file_path = os.path.join(DATA_PATH, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
                print(f"✅ All files in '{DATA_PATH}' have been deleted.")
            else:
                print(f"ℹ️  Folder '{DATA_PATH}' does not exist.")
        else:
            print("👌 Source files kept safe.")

    else:
        print("❌ Operation cancelled.")

if __name__ == "__main__":
    main()