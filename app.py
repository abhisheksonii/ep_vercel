import os
import base64
import json
import logging
import anthropic
import tempfile
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, send_file, session
from functools import wraps
from io import BytesIO
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path

# Load environment variables
load_dotenv()

def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
    
    # Use /tmp directory for file uploads on Render
    UPLOAD_FOLDER = tempfile.gettempdir()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'png', 'jpeg'}
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    USERNAME = "epgroup"
    PASSWORD = "epgroup123"
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    class PDFProcessingError(Exception):
        pass

    class PDFVisionProcessor:
        def __init__(self):
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise PDFProcessingError("ANTHROPIC_API_KEY not found in environment variables")

            try:
                self.client = anthropic.Anthropic(
                    api_key=api_key
                )
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {str(e)}")
                raise PDFProcessingError(f"Failed to initialize Anthropic client: {str(e)}")

        def check_and_correct_orientation(self, image):
            """
            Check if image is horizontal and rotate if needed to make it vertical.
            Returns the correctly oriented image.
            """
            try:
                width, height = image.size
                
                # If width is greater than height, it's horizontal
                if width > height:
                    # Rotate 90 degrees clockwise to make it vertical
                    image = image.rotate(-90, expand=True)
                    logger.info("Image was horizontal, rotating to vertical orientation")
                
                return image
            except Exception as e:
                logger.error(f"Error checking/correcting image orientation: {str(e)}")
                raise PDFProcessingError(f"Failed to process image orientation: {str(e)}")

        def validate_json_structure(self, data):
            required_fields = [
                "inbound_delivery_no", "shipment_no", "shipping_cost_doc_no",
                "vendor", "vendor_name", "vessel_name", "eta_date",
                "port_arrival", "port_departure", "shipping_line",
                "bill_of_lading", "invoice_no", "invoice_date",
                "invoice_value", "mode_of_pay", "lc_no", "due_date",
                "bank_ref_no", "container_numbers", "no_of_cartons",
                "gross_weight", "product_details"
            ]
            
            product_fields = [
                "item", "sap_code", "description", "tariff_code",
                "quantity_uom", "packaging", "po_no", "item_no"
            ]

            # Check for required top-level fields
            for field in required_fields:
                if field not in data:
                    data[field] = ""  # Set empty string for missing fields

            # Ensure product_details is a list
            if not isinstance(data.get("product_details"), list):
                data["product_details"] = []

            # Validate each product in product_details
            for product in data["product_details"]:
                for field in product_fields:
                    if field not in product:
                        product[field] = ""  # Set empty string for missing fields

            return data

        def extract_json_from_text(self, text):
            try:
                # Try to find JSON object in the text
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                
                if start_idx == -1 or end_idx == -1:
                    raise PDFProcessingError("No JSON object found in response")
                
                json_str = text[start_idx:end_idx + 1]
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                logger.error(f"Raw text: {text}")
                raise PDFProcessingError("Failed to parse JSON from response")

        def process_with_claude(self, image_base64):
            if not image_base64:
                raise PDFProcessingError("No image data provided")

            try:
                message = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4096,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": """Extract all the information from this image and format it ONLY as a JSON object with the following structure - DO NOT include any other text or explanation:
{
    "inbound_delivery_no": "",
    "shipment_no": "",
    "shipping_cost_doc_no": "",
    "vendor": "",
    "vendor_name": "",
    "vessel_name": "",
    "eta_date": "",
    "port_arrival": "",
    "port_departure": "",
    "shipping_line": "",
    "bill_of_lading": "",
    "invoice_no": "",
    "invoice_date": "",
    "invoice_value": "",
    "mode_of_pay": "",
    "lc_no": "",
    "due_date": "",
    "bank_ref_no": "",
    "container_numbers": "",
    "no_of_cartons": "",
    "gross_weight": "",
    "product_details": [
        {
            "item": "",
            "sap_code": "",
            "description": "",
            "tariff_code": "",
            "quantity_uom": "",
            "packaging": "",
            "po_no": "",
            "item_no": ""
        }
    ]
}"""
                            }
                        ]   
                    }]
                )

                if not hasattr(message, 'content') or not message.content:
                    raise PDFProcessingError("Empty response from Claude")

                for content in message.content:
                    if hasattr(content, 'text'):
                        try:
                            # Extract JSON from text and validate structure
                            extracted_data = self.extract_json_from_text(content.text)
                            validated_data = self.validate_json_structure(extracted_data)
                            return validated_data
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {str(e)}")
                            logger.error(f"Raw text content: {content.text}")
                            raise PDFProcessingError(f"Failed to parse JSON response: {str(e)}")

                raise PDFProcessingError("No valid text content found in Claude's response")

            except Exception as e:
                logger.error(f"Claude Vision API processing failed: {str(e)}")
                raise PDFProcessingError(f"Failed to process with Claude Vision API: {str(e)}")

        def convert_file_to_images(self, file_path):
            """Convert PDF or image file to list of images with orientation correction"""
            if not os.path.exists(file_path):
                raise PDFProcessingError("File not found")

            try:
                images = []
                if file_path.endswith('.pdf'):
                    # Convert PDF to images
                    pdf_images = convert_from_path(file_path)
                    # Check and correct orientation for each PDF page
                    for image in pdf_images:
                        corrected_image = self.check_and_correct_orientation(image)
                        images.append(corrected_image)
                else:
                    # Handle regular image files
                    image = Image.open(file_path)
                    # Check and correct orientation
                    corrected_image = self.check_and_correct_orientation(image)
                    images = [corrected_image]

                if not images:
                    raise PDFProcessingError("No images extracted from file")
                return images
            except Exception as e:
                logger.error(f"File to image conversion failed: {str(e)}")
                raise PDFProcessingError(f"Failed to convert file to images: {str(e)}")

        def encode_image_to_base64(self, image):
            """Convert PIL Image to base64 string"""
            if not isinstance(image, Image.Image):
                raise PDFProcessingError("Invalid image format")

            try:
                buffered = BytesIO()
                # Convert to RGB if image is in RGBA mode
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image.save(buffered, format="JPEG", quality=95)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception as e:
                logger.error(f"Image encoding failed: {str(e)}")
                raise PDFProcessingError(f"Failed to encode image to base64: {str(e)}")

    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'logged_in' not in session:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            if username == USERNAME and password == PASSWORD:
                session['logged_in'] = True
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password')
                return redirect(url_for('login'))
        
        return render_template('login.html')

    @app.route('/logout')
    def logout():
        session.pop('logged_in', None)
        return redirect(url_for('login'))

    @app.route('/')
    @login_required
    def index():
        return render_template('index.html', data=None)

    @app.route('/upload', methods=['POST'])
    @login_required
    def upload_file():
        if 'files' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files')

        if not any(file.filename for file in files):
            flash('No selected files')
            return redirect(request.url)

        all_data = []
        temp_files = []  # Keep track of temporary files

        try:
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    
                    # Create a temporary file
                    temp_fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(filename)[1])
                    temp_files.append(temp_path)  # Add to list for cleanup
                    
                    try:
                        # Save uploaded file to temporary location
                        with os.fdopen(temp_fd, 'wb') as tmp:
                            file.save(tmp)
                        
                        processor = PDFVisionProcessor()
                        images = processor.convert_file_to_images(temp_path)

                        for image in images:
                            image_base64 = processor.encode_image_to_base64(image)
                            extracted_data = processor.process_with_claude(image_base64)
                            all_data.append(extracted_data)

                    except Exception as e:
                        logger.error(f"Processing error for file {filename}: {str(e)}")
                        flash(f"Error processing {filename}: {str(e)}")
                        return redirect(url_for('index'))
                else:
                    flash(f"Invalid file type: {file.filename}")
                    return redirect(url_for('index'))

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.error(f"Error deleting temporary file {temp_file}: {str(e)}")

        return render_template('index.html', data=all_data[0] if all_data else None)

    @app.route('/save_report', methods=['POST'])
    @login_required
    def save_report():
        try:
            report_data = request.json
            if not report_data:
                return jsonify({"error": "No data provided"}), 400
            
            # Generate the HTML report with the updated data
            report_html = render_template('report.html', data=report_data)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
                temp_file.write(report_html.encode())
                
            return jsonify({"message": "Report saved successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/download_report', methods=['POST'])
    @login_required
    def download_report():
        try:
            report_data = request.json
            if not report_data:
                return jsonify({"error": "No data provided"}), 400

            # Generate the HTML report
            report_html = render_template('report.html', data=report_data)

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
                temp_file.write(report_html.encode())
                temp_file.flush()

                # Return the file for download
                return send_file(
                    temp_file.name,
                    as_attachment=True,
                    download_name='shipment_report.html',
                    mimetype='text/html'
                )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
