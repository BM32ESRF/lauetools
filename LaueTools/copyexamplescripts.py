import os
import shutil

def copy_examples(source_dir='examples', target_subdir='copied_files', overwrite=False, allowed_extensions=('.py', '.ipynb','.pdf'), verbose=False):
    # Ensure source directory exists
    if not os.path.isdir(source_dir):
        print(f"The source folder '{source_dir}' does not exist.")
        return

    # Create target directory as a subdirectory of the current working directory
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, target_subdir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target subdirectory: {target_dir}")

    print('copying only files with the following extensions: ', allowed_extensions)
    # Copy files with allowed extensions
    for filename in os.listdir(source_dir):
        src = os.path.join(source_dir, filename)
        dst = os.path.join(target_dir, filename)

        if os.path.isfile(src):
            if filename.endswith(allowed_extensions):
                if not os.path.exists(dst) or overwrite:
                    shutil.copy2(src, dst)
                    print(f"Copied: {filename}")
                else:
                    print(f"Already exists, skipped: {filename}")
            else:
                if verbose:
                    print(f"Skipped (unsupported extension): {filename}")
        else:
            print(f"Skipped (not a file): {filename}")

    return target_dir

def start():
    import LaueTools as LT
    import LaueTools.generaltools as GT
    lauetoolsmainfolder = os.path.dirname(os.path.abspath(LT.__file__))
    target_dir = copy_examples(source_dir=os.path.join(lauetoolsmainfolder, 'notebooks'),
                  target_subdir='lauetools_scripts',
                  overwrite=False)
    if target_dir:
        GT.printgreen("Done.")
        GT.printgreen(f"Scripts were copied to {target_dir} folder.")

if __name__ == "__main__":
    start() 
    