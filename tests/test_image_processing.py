#!/usr/bin/env python3
"""
Test script for image processing functionality.
Enhanced with thumbnail generation for gallery system.
"""

import sys
import os
import re
import hashlib
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image
import io

def generate_image_thumbnail(image_path: Path, thumbnail_path: Path, size: tuple = (200, 200)) -> bool:
    """
    Generate a thumbnail for an image.

    Args:
        image_path: Path to the original image
        thumbnail_path: Path where to save the thumbnail
        size: Maximum size for the thumbnail (width, height)

    Returns:
        True if thumbnail was generated successfully, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for JPEG compatibility)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # Create thumbnail maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)

            # Save thumbnail
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
            return True
    except Exception as e:
        print(f"Failed to generate thumbnail for {image_path}: {e}")
        return False

def process_html_images(html_content: str) -> str:
    """
    Process HTML content to download external images and replace URLs with local paths.

    Args:
        html_content: HTML content containing potentially external image URLs

    Returns:
        HTML content with local image URLs
    """
    try:
        # Find all img tags with src attributes
        img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
        matches = re.findall(img_pattern, html_content, re.IGNORECASE)

        if not matches:
            return html_content

        # Create images directory if it doesn't exist
        images_dir = Path("src/ui/assets/images")
        images_dir.mkdir(parents=True, exist_ok=True)

        # Process each image URL
        processed_urls = {}
        for url in matches:
            if url in processed_urls:
                continue  # Already processed this URL

            try:
                # Check if it's an external URL or a local file URL that needs processing
                needs_processing = url.startswith(('http://', 'https://'))

                # Also process local file:// URLs if they don't have thumbnails yet
                if url.startswith('file://'):
                    # Convert file:// URL to local path
                    try:
                        from urllib.parse import unquote
                        file_path_str = url[7:]  # Remove 'file://' prefix
                        file_path_str = unquote(file_path_str)  # Decode URL encoding
                        local_file_path = Path(file_path_str)

                        # Check if this is an image that needs thumbnail processing
                        if local_file_path.exists() and local_file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                            # Check if thumbnail exists
                            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                            thumbnail_filename = f"{url_hash}_thumb.jpg"
                            thumbnail_path = images_dir / thumbnail_filename

                            if not thumbnail_path.exists():
                                needs_processing = True
                                print(f"Processing local image for thumbnail: {local_file_path}")
                            else:
                                print(f"Thumbnail already exists for: {local_file_path}")
                    except Exception as e:
                        print(f"Error checking local file: {e}")

                if needs_processing:
                    if url.startswith('file://'):
                        # Handle local file processing
                        from urllib.parse import unquote
                        file_path_str = url[7:]  # Remove 'file://' prefix
                        file_path_str = unquote(file_path_str)  # Decode URL encoding
                        local_file_path = Path(file_path_str)

                        # Generate hash for this local file
                        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                        thumbnail_filename = f"{url_hash}_thumb.jpg"
                        thumbnail_path = images_dir / thumbnail_filename

                        # Generate thumbnail from the local file
                        if generate_image_thumbnail(local_file_path, thumbnail_path):
                            print(f"Generated thumbnail for local file: {thumbnail_path}")

                            # Use thumbnail URL for gallery, original file URL for modal
                            thumbnail_url = f"file://{thumbnail_path.resolve()}"
                            full_image_url = url  # Keep original file:// URL

                            # Store both URLs (thumbnail for display, full for modal)
                            processed_urls[url] = {
                                'thumbnail': thumbnail_url,
                                'full': full_image_url
                            }
                        else:
                            print(f"Failed to generate thumbnail for local file: {local_file_path}")
                            processed_urls[url] = url  # Keep original URL
                    else:
                        # Handle external URL processing
                        # Generate filename from URL hash to avoid conflicts
                        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                        # Try to get file extension from URL
                        url_path = urllib.parse.urlparse(url).path
                        ext = Path(url_path).suffix.lower()
                        if not ext or ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                            ext = '.jpg'  # Default extension

                        filename = f"{url_hash}{ext}"
                        local_path = images_dir / filename

                        # Create thumbnail path
                        thumbnail_filename = f"{url_hash}_thumb.jpg"
                        thumbnail_path = images_dir / thumbnail_filename

                        # Check if we already downloaded this image
                        if not local_path.exists():
                            print(f"Downloading image: {url}")
                            # Download the image
                            with urllib.request.urlopen(url, timeout=10) as response:
                                if response.status == 200:
                                    image_data = response.read()
                                    with open(local_path, 'wb') as f:
                                        f.write(image_data)
                                    print(f"Downloaded image to: {local_path}")

                                    # Generate thumbnail
                                    if generate_image_thumbnail(local_path, thumbnail_path):
                                        print(f"Generated thumbnail: {thumbnail_path}")
                                    else:
                                        print(f"Failed to generate thumbnail for: {local_path}")
                                else:
                                    print(f"Failed to download image {url}: HTTP {response.status}")
                                    processed_urls[url] = url  # Keep original URL
                                    continue
                        else:
                            print(f"Image already exists: {local_path}")
                            # Generate thumbnail if it doesn't exist
                            if not thumbnail_path.exists():
                                if generate_image_thumbnail(local_path, thumbnail_path):
                                    print(f"Generated missing thumbnail: {thumbnail_path}")

                        # Use thumbnail URL for gallery, full image for modal
                        thumbnail_url = f"file://{thumbnail_path.resolve()}"
                        full_image_url = f"file://{local_path.resolve()}"

                        # Store both URLs (thumbnail for display, full for modal)
                        processed_urls[url] = {
                            'thumbnail': thumbnail_url,
                            'full': full_image_url
                        }
                else:
                    # Already local or relative URL, keep as is
                    processed_urls[url] = url

            except Exception as e:
                print(f"Failed to process image URL {url}: {e}")
                processed_urls[url] = url  # Keep original URL on error

        # Replace URLs in HTML - now we use thumbnail URLs for display
        processed_html = html_content
        for original_url, url_data in processed_urls.items():
            if isinstance(url_data, dict):
                # Use thumbnail URL for display, store full URL in data attribute
                thumbnail_url = url_data['thumbnail']
                full_url = url_data['full']

                # For file:// URLs with backslashes, we need special handling
                if original_url.startswith('file://'):
                    # Use a more direct approach for file URLs
                    processed_html = processed_html.replace(
                        f'src="{original_url}"',
                        f'src="{thumbnail_url}" data-full-src="{full_url}"'
                    )
                    processed_html = processed_html.replace(
                        f"src='{original_url}'",
                        f"src='{thumbnail_url}' data-full-src='{full_url}'"
                    )
                else:
                    # Use regex for other URLs
                    escaped_url = re.escape(original_url)
                    processed_html = re.sub(
                        rf'<img([^>]+)src=["\']{escaped_url}["\']([^>]*)>',
                        f'<img\\1src="{thumbnail_url}" data-full-src="{full_url}"\\2>',
                        processed_html,
                        flags=re.IGNORECASE
                    )
            else:
                # Fallback for non-processed URLs
                if original_url.startswith('file://'):
                    # Direct replacement for file URLs
                    processed_html = processed_html.replace(
                        f'src="{original_url}"',
                        f'src="{url_data}"'
                    )
                    processed_html = processed_html.replace(
                        f"src='{original_url}'",
                        f"src='{url_data}'"
                    )
                else:
                    # Use regex for other URLs
                    escaped_url = re.escape(original_url)
                    processed_html = re.sub(
                        rf'src=["\']{escaped_url}["\']',
                        f'src="{url_data}"',
                        processed_html,
                        flags=re.IGNORECASE
                    )

        return processed_html

    except Exception as e:
        print(f"Error processing HTML images: {e}")
        return html_content  # Return original content on error

def create_test_image():
    """Create a simple test image for testing thumbnail generation."""
    from PIL import Image, ImageDraw

    # Create a simple test image
    img = Image.new('RGB', (400, 300), color='blue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 350, 250], fill='red')
    draw.text((200, 150), "Test Image", fill='white')

    # Save test image
    images_dir = Path("src/ui/assets/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    test_image_path = images_dir / "test_image.jpg"
    img.save(test_image_path, 'JPEG')
    return test_image_path

def test_thumbnail_generation():
    """Test thumbnail generation functionality."""
    print("Testing thumbnail generation...")

    # Create a test image
    test_image_path = create_test_image()
    print(f"Created test image: {test_image_path}")

    # Generate thumbnail
    thumbnail_path = test_image_path.parent / "test_thumb.jpg"
    success = generate_image_thumbnail(test_image_path, thumbnail_path)

    if success:
        print(f"Thumbnail generated successfully: {thumbnail_path}")

        # Check file sizes
        original_size = test_image_path.stat().st_size
        thumbnail_size = thumbnail_path.stat().st_size
        print(f"Original size: {original_size} bytes")
        print(f"Thumbnail size: {thumbnail_size} bytes")
        print(f"Compression ratio: {thumbnail_size/original_size:.2f}")
    else:
        print("Thumbnail generation failed")

    return success

def test_image_processing():
    """Test the image processing functionality."""
    print("\nTesting image processing...")

    # Create a test image first
    test_image_path = create_test_image()
    test_image_url = f"file://{test_image_path.resolve()}"

    # Test HTML with local image
    test_html = f'''
    <p>Here are some test images:</p>
    <img src="{test_image_url}" alt="Test Image 1">
    <p>End of test.</p>
    '''

    print("Original HTML:")
    print(test_html)

    processed_html = process_html_images(test_html)

    print("\nProcessed HTML:")
    print(processed_html)

    # Check if images directory exists and has files
    images_dir = Path("src/ui/assets/images")
    if images_dir.exists():
        files = list(images_dir.glob("*"))
        print(f"\nImages directory contains {len(files)} files:")
        for file in files:
            print(f"  - {file.name} ({file.stat().st_size} bytes)")
    else:
        print("\nImages directory does not exist")

if __name__ == "__main__":
    # Test thumbnail generation first
    test_thumbnail_generation()

    # Then test image processing
    test_image_processing()
