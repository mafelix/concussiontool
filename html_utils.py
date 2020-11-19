"""
This is a method that converts a list to an html unordered list.
input: list of strings
outputs: html list
"""
def ToHtmlList(list_obj, style="margin-left: 2em;"):
    html_string = '<ul style="' + style + '">'
    for list_item_index in range(len(list_obj)):
        html_string += '<li>' + list_obj[list_item_index] + '</li>'
    html_string += '</ul>'
    return html_string