# SmartDown

SmartDown is a Python project that extracts main content from HTML articles and converts them to Markdown format.

## Installation

Make sure you have Python 3.11 or later installed. Then, you can install the dependencies using [Poetry](https://python-poetry.org/):

```shell
poetry install
```

## Usage
To use SmartDown, you can run the main script smartdown.py with the following command:

```bash
python smartdown.py <input_file.html> <output_file.md>
```

Replace `<input_file.html>` with the path to the HTML file you want to convert, and `<output_file.md>` with the desired path for the resulting Markdown file.

### Dependencies
SmartDown relies on the following Python libraries:

* spacy (version 3.5.3)
* beautifulsoup4 (version 4.12.2)
* markdownify (version 0.11.6)
* markdown (version 3.4.3)
You can install these dependencies by running:

```bash
poetry install
```
License
SmartDown is licensed under the MIT License. See the `LICENSE` file for more information.

Contact
If you have any questions or suggestions, feel free to reach out to the project author:

* Santiago Silva
* Email: 16252054+sssilvar@users.noreply.github.com
