import pandas as pd
import xml.etree.ElementTree as ET

# Define a function to parse XML and extract stats
def extract_xml_stats(xml_column):
    def parse_xml(xml_data):
        if pd.isnull(xml_data) or xml_data.strip() == '<goal/>':  
            return 0 

        try:
            # Parse the XML data
            root = ET.fromstring(xml_data)
            # Count the number of 'value' elements, each represents an event
            return len(root.findall('value'))
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return None  # Return None or 0 if there is a parsing error

    return match_df[xml_column].apply(parse_xml)

def extract (match):
    print('Extracting XML Data...')
    global match_df
    match_df = match
    # Extract stats for each XML-containing column
    match_df['goal_count'] = extract_xml_stats('goal')
    match_df['shoton_count'] = extract_xml_stats('shoton')
    match_df['shotoff_count'] = extract_xml_stats('shotoff')
    match_df['foulcommit_count'] = extract_xml_stats('foulcommit')
    match_df['card_count'] = extract_xml_stats('card')
    match_df['cross_count'] = extract_xml_stats('cross')
    match_df['corner_count'] = extract_xml_stats('corner')
    match_df['possession_count'] = extract_xml_stats('possession')

    # Now matches_df has new columns for each type of event count
    #print(match_df[['goal_count', 'shoton_count', 'shotoff_count', 'foulcommit_count', 'card_count', 'cross_count', 'corner_count', 'possession_count']])

    for column in ['goal', 'shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner', 'possession']:
        match_df[[f'{column}_elapsed', f'{column}_player1', f'{column}_team', f'{column}_type', f'{column}_detail']] = match_df[column].apply(lambda x: extract_detailed_stats(x, column))

    print('Finished Extracting XML Data')
    return match_df
# This code will create new columns in the matches_df DataFrame, each representing the count of different event types extracted from the XML data. ###

from lxml import etree


# Define a function to parse XML for detailed extraction
def extract_detailed_stats(xml_data, tag):
    if pd.isnull(xml_data) or xml_data.strip() == f'<{tag}/>':  # Check if XML data is empty
        return pd.Series([None]*5)  # Adjust the number based on the data points you expect

    try:
        # Parse the XML data
        root = etree.fromstring(xml_data)
        
        # Initialize lists to store the extracted data
        elapsed_list = []
        player1_list = []
        team_list = []
        type_list = []
        detail_list = []

        # Extract information for each 'value' element within the tag
        for value in root.xpath('.//value'):
            elapsed_list.append(value.findtext('elapsed'))
            player1_list.append(value.findtext('player1'))
            team_list.append(value.findtext('team'))
            type_list.append(value.findtext('type'))
            detail_list.append(value.findtext('comment'))  # This can be adjusted to other details

        return pd.Series([elapsed_list, player1_list, team_list, type_list, detail_list])

    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML: {e}")
        return pd.Series([None]*5)  # Adjust the number based on the data points you expect

# Extract detailed stats for each XML-containing column

# Now matches_df has new columns for each detail extracted from the XML data
#print(match_df.head())


#This code goes through each XML column and extracts the time of the event (elapsed), the player involved (player1), 
# the team involved (team), the type of event (type), and an additional detail (comment).
#The extract_detailed_stats function returns a pandas Series object with lists for each type of detail. 
# These lists are then assigned as new columns in the matches_df DataFrame. For instance, you will have goal_elapsed,
# goal_player1, goal_team, goal_type, goal_detail, and similar columns for shoton, shotoff, foulcommit, etc.
