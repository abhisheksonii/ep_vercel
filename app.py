import os
import base64
import json
import logging
import fitz  # PyMuPDF
import anthropic
import tempfile
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, send_file
from io import BytesIO
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'png', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class PDFProcessingError(Exception):
    pass

class PDFVisionProcessor:
    def __init__(self):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise PDFProcessingError("ANTHROPIC_API_KEY not found in environment variables")

        try:
            self.client = anthropic.Client(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            raise PDFProcessingError(f"Failed to initialize Anthropic client: {str(e)}")

    def convert_file_to_images(self, file_path):
        if not os.path.exists(file_path):
            raise PDFProcessingError("File not found")

        try:
            images = []
            if file_path.endswith('.pdf'):
                pdf_document = fitz.open(file_path)
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    pix = page.get_pixmap()
                    img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img_data)
                pdf_document.close()
            else:
                image = Image.open(file_path)
                images = [image]

            if not images:
                raise PDFProcessingError("No images extracted from file")
            return images
        except Exception as e:
            logger.error(f"File to image conversion failed: {str(e)}")
            raise PDFProcessingError(f"Failed to convert file to images: {str(e)}")

    def encode_image_to_base64(self, image):
        if not isinstance(image, Image.Image):
            raise PDFProcessingError("Invalid image format")

        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Image encoding failed: {str(e)}")
            raise PDFProcessingError(f"Failed to encode image to base64: {str(e)}")

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
                            "text": """Extract all the information from this image and format it as a JSON object with the following structure:
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

            if hasattr(message, 'content') and isinstance(message.content, list):
                for content in message.content:
                    if hasattr(content, 'text'):
                        try:
                            extracted_data = json.loads(content.text)
                            return self.map_data_to_structure(extracted_data)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {str(e)}")
                            logger.error(f"Raw text content: {content.text}")
                            raise PDFProcessingError(f"Failed to parse JSON response: {str(e)}")
                    
            raise PDFProcessingError("No valid text content found in Claude's response")
            
        except Exception as e:
            logger.error(f"Claude Vision API processing failed: {str(e)}")
            raise PDFProcessingError(f"Failed to process with Claude Vision API: {str(e)}")

    def map_data_to_structure(self, extracted_data):
        structured_data = {
            "inbound_delivery_no": extracted_data.get("inbound_delivery_no", ""),
            "shipment_no": extracted_data.get("shipment_no", ""),
            "shipping_cost_doc_no": extracted_data.get("shipping_cost_doc_no", ""),
            "vendor": extracted_data.get("vendor", ""),
            "vendor_name": extracted_data.get("vendor_name", ""),
            "vessel_name": extracted_data.get("vessel_name", ""),
            "eta_date": extracted_data.get("eta_date", ""),
            "port_arrival": extracted_data.get("port_arrival", ""),
            "port_departure": extracted_data.get("port_departure", ""),
            "shipping_line": extracted_data.get("shipping_line", ""),
            "bill_of_lading": extracted_data.get("bill_of_lading", ""),
            "invoice_no": extracted_data.get("invoice_no", ""),
            "invoice_date": extracted_data.get("invoice_date", ""),
            "invoice_value": extracted_data.get("invoice_value", ""),
            "mode_of_pay": extracted_data.get("mode_of_pay", ""),
            "lc_no": extracted_data.get("lc_no", ""),
            "due_date": extracted_data.get("due_date", ""),
            "bank_ref_no": extracted_data.get("bank_ref_no", ""),
            "container_numbers": extracted_data.get("container_numbers", ""),
            "no_of_cartons": extracted_data.get("no_of_cartons", ""),
            "gross_weight": extracted_data.get("gross_weight", ""),
            "product_details": extracted_data.get("product_details", [])
        }
        return structured_data

@app.route('/')
def index():
    return render_template('index.html', data=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('files')

    if not any(file.filename for file in files):
        flash('No selected files')
        return redirect(request.url)

    all_data = []
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                processor = PDFVisionProcessor()
                images = processor.convert_file_to_images(filepath)

                for image in images:
                    image_base64 = processor.encode_image_to_base64(image)
                    extracted_data = processor.process_with_claude(image_base64)
                    all_data.append(extracted_data)

                # Cleanup
                os.remove(filepath)
            except PDFProcessingError as e:
                flash(str(e))
                return redirect(url_for('index'))
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                flash(f"An unexpected error occurred: {str(e)}")
                return redirect(url_for('index'))
        else:
            flash(f"Invalid file type: {file.filename}")
            return redirect(url_for('index'))

    return render_template('index.html', data=all_data[0] if all_data else None)

@app.route('/save_report', methods=['POST'])
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

if __name__ == '__main__':
    app.run(debug=True)