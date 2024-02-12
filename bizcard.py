
import cv2
import pytesseract
import pandas as pd
import re
import mysql.connector 
import streamlit as st
import numpy as np
from PIL import Image


# Set the path to the Tesseract executable (if not set in system environment variables)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Function to extract information from image and display in Streamlit
def process_image(image_data):
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    extracted_text = extract_text_from_image(image)
    lines = extracted_text.split('\n')
    # Your existing code for processing the extracted text
    return lines

# Function to extract text from an image using OCR
def extract_text_from_image(image):
    # Load the image using OpenCV
    #image = cv2.imread(image_path)
    
    # Resize the image to improve OCR accuracy
    #resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Convert the resized image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # Use pytesseract to perform OCR on the grayscale image
    extracted_text = pytesseract.image_to_string(gray_image)
    
    return extracted_text

def process_image(image):
    # Extract text from the image
    extracted_text = extract_text_from_image(image)

    # Split the text into lines
    lines = extracted_text.split('\n')

    Text = []
    for i in lines: 
        Text.append(i)

    #print(Text)
    final_list = []
    # Extract company name
    def extract_company_names(Text):
        # Regular expression pattern to match capitalized words followed by indicators of a company
        pattern = r'\b[A-Z][a-zA-Z\s]+(?:LLC|Inc\.|Corp\.|Ltd\.|Company)\b'
        
        company_names = [] 
        for text in Text:
            # Find all matches of the pattern in the current string
            matches = re.findall(pattern, text)
            # Add the matches to the list of company names
            company_names.extend(matches)
        
        return company_names

    # Call the function to extract company names
    company_names = extract_company_names(Text)
    if len(company_names) == 0:
        # Iterate through the list, skipping the first and last elements
        middle_elements = [Text[i+1] for i in range(len(Text)-1) if Text[i] == '']
        final_list.append(middle_elements[0])
    # print(middle_elements[0])
    else:
        # Display the list of extracted company names
        final_list.append(middle_elements)
    #  print(company_names)

    # Extract phone numbers from the list
    def extract_phone_numbers(text_list):
        # Regular expression pattern to match phone numbers
        phone_pattern = r'\b\d{9}m\b|\b\d{10}\b|\b\d{10,}\b|\b(?:\d[ -]?){9,10}\d(?:m)?\b'  # Matches sequences of 10 digits followed by 'm', sequences of 10 digits, or sequences of more than 10 digits
        #'\b(?:\d[ -]?){9,10}\d(?:m)?\b'  # Matches sequences of digits optionally separated by spaces or hyphens, with optional 'm' at the end
        
        # Initialize an empty list to store the extracted phone numbers
        phone_numbers = []
        
        # Iterate through each string in the list
        for text in text_list:
            # Find all phone numbers that match the pattern in the current string
            matches = re.findall(phone_pattern, text)
            
            # Add the matched phone numbers to the list
            phone_numbers.extend(matches)
        
        return phone_numbers 

    # Extract phone numbers from the list
    phone_numbers = extract_phone_numbers(Text)
    
    if len(phone_numbers) == 2:
        final_list.append(phone_numbers[0]+","+phone_numbers[1])
    #   print(phone_numbers[0]+","+phone_numbers[1])
    else:
        # Print the extracted phone numbers
        final_list.append(phone_numbers)
    #    print(phone_numbers)

    # Function to validate website URLs
    def validate_website(url):
        # Regular expression pattern to match a valid website URL
        url_pattern = r'(https?://)?(www\.)?[\w.-]+\.[a-zA-Z]{2,}(/\S*)?'
        
        # Check if the string matches the URL pattern
        if re.match(url_pattern, url):
            return url
        else:
            return None

    # Extract valid website URLs
    websites = [validate_website(Text) for Text in Text if validate_website(Text)]

    final_list.append(websites)
    # Display the validated websites
    #print(websites)

    # Function to extract email IDs from a list and handle potential mistakes
    def extract_and_validate_emails(Text):
        valid_emails = []
        potential_emails = []
        
        for email in Text:
            # Use regular expression to check for email format
            if re.match(r"[^@]+@[^@]+\.[^@]+", email):
                # Valid email format
                valid_emails.append(email)
            elif "@" in email:
                # If "@" is present but email format is not valid, treat it as a potential email
                potential_emails.append(email)
            else:
                # Other cases can be treated as potential mistakes
                pass
        
        return valid_emails, potential_emails

    # Extract and validate emails
    valid_emails, potential_emails = extract_and_validate_emails(Text)

    # If there are no valid emails, treat potential emails as email IDs
    if not valid_emails and potential_emails:
        valid_emails.extend(potential_emails)

    final_list.append(valid_emails)
    # Display the valid email IDs
    #print(*valid_emails) 

    # Function to extract person names from a list of strings
    def extract_person_names(text_list):
        # Define patterns or rules for identifying person names
        person_names = []
        for text in text_list:
            # Tokenize the text and iterate through each word
            for word in text.split():
                # Check if the word starts with a capital letter and is not an email address or website URL
                if word.istitle() and not re.match(r'^[\w\.-]+@[\w\.-]+$', word) and not re.match(r'^https?://\S+$', word):
                    person_names.append(word)
        
        return person_names

    # Extract person names from the list
    person_names = extract_person_names(Text)

    if len(person_names) > 3:
        final_list.append(person_names[0]+"/"+person_names[1])
        #print(person_names[0]+" "+person_names[1])
    else:
        # Print the extracted person names
        final_list.append(person_names)
        #print(*person_names)

    # Function to extract job titles/positions from the list
    def extract_job_titles(Text):
        job_titles = []
        for text in Text:
            # Check if the text contains only capital letters and no special characters
            if text.isupper() and not re.search(r'[^\w\s]', text):
                job_titles.append(text)
        return job_titles

    # Extract job titles/positions from the list
    job_titles = extract_job_titles(Text)

    final_list.append(job_titles)
    # Print the extracted job titles
    #print(*job_titles)

    # Function to extract address from text
    def extract_address(Text):
        # Join the text lines into a single string for better pattern matching
        text = ' '.join(Text)
        
        # Define regex pattern to match addresses
        address_pattern = r'\b\d+\s+[a-zA-Z0-9\s,]+\b'
        
        # Find all matches of the address pattern in the text
        addresses = re.findall(address_pattern, text)
        
        # Return the list of addresses
        return addresses


    # Extract address from the text
    addresses = extract_address(Text)
    final_list.append(addresses)

    Final_list = [
        ['Company Name', 'Phone Numbers', 'Website', 'Emails', 'Person Name', 'Job Title', 'Address'],final_list]

    # Replace empty strings with NaN (missing value indicator)
    final_list_with_nan = [[value if value else pd.NA for value in sublist] for sublist in Final_list]

    # Convert final_list_with_nan to DataFrame
    df = pd.DataFrame(final_list_with_nan[1:], columns=final_list_with_nan[0])

    return df

# Streamlit App
def main():
    st.title("Image Information Extraction and Editing App")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Extracting information from the image...")
        lines = process_image(image)
        
        # Display the lines of text extracted from the image as a table
        st.write("Lines of Text Extracted from Image (in table format):")
        st.table(lines)
      
        
        st.write("")
        st.write("Extracted Information:")
        df = process_image(image)  # Define df here
        # Display the DataFrame containing extracted information
        st.write(df)

        # Edit DataFrame values
        st.write("Edit DataFrame Values:")
        for column in df.columns:
            edited_values = st.text_input(f"Edit values for '{column}':", value=df[column].fillna('').values, key=column)
            df[column] = edited_values

        # Button to confirm editing and transfer data to MySQL
        if st.button("Confirm Editing and Transfer Data to MySQL"):
            # Connect to MySQL database
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="Pass@12345678",
                database="bussiness_card_database"
            )

            try:
                cursor = conn.cursor()

                # Loop through columns and create a table with dynamic SQL
                create_table_query = f"CREATE TABLE IF NOT EXISTS card_details ("
                for col_name, dtype in df.dtypes.items():
                    mysql_dtype = "TEXT"  # Default to TEXT type
                    if dtype == "int64":
                        mysql_dtype = "INT"
                    elif dtype == "float64":
                        mysql_dtype = "FLOAT"
                    create_table_query += f"`{col_name}` {mysql_dtype}, "
                create_table_query = create_table_query[:-2] + ")"  # Remove the last comma
                cursor.execute(create_table_query)

                # Insert data into the table
                for i, row in df.iterrows():
                    insert_query = f"INSERT INTO card_details (`{str.join('`, `', df.columns)}`) VALUES ({', '.join(['%s'] * len(df.columns))})"
                    # Convert row data to tuple of strings
                    row_data = tuple(str(value) for value in row)
                    cursor.execute(insert_query, row_data)
                conn.commit()
                st.success("Data transferred to MySQL successfully!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}") 
            finally:
                cursor.close()
                conn.close()


if __name__ == "__main__":
    main()

