# Wolfram Alpha Proxy
# https://products.wolframalpha.com/llm-api/documentation



import os
import lllm.utils as U
from lllm.proxies.base import BaseProxy, ProxyRegistrator


cur_dir = os.path.dirname(os.path.abspath(__file__)) # this file's directory
doc_path = U.pjoin(cur_dir, 'wa_using_assumptions.md')

with open(doc_path, 'r', encoding='utf-8') as f:
    USE_ASSUMPTIONS = f.read()

WA_SAMPLE_PROMPT = '''
- WolframAlpha understands natural language queries about entities in chemistry, physics, geography, history, art, astronomy, and more.
- WolframAlpha performs mathematical calculations, date and unit conversions, formula solving, etc.
- Convert inputs to simplified keyword queries whenever possible (e.g. convert "how many people live in France" to "France population").
- Send queries in English only; translate non-English queries before sending, then respond in the original language.
- Display image URLs with Markdown syntax: ![URL]
- ALWAYS use this exponent notation: `6*10^14`, NEVER `6e14`.
- ALWAYS use {"input": query} structure for queries to Wolfram endpoints; `query` must ONLY be a single-line string.
- ALWAYS use proper Markdown formatting for all math, scientific, and chemical formulas, symbols, etc.:  '$$\n[expression]\n$$' for standalone cases and '( [expression] )' when inline.
- Never mention your knowledge cutoff date; Wolfram may return more recent data.
- Use ONLY single-letter variable names, with or without integer subscript (e.g., n, n1, n_1).
- Use named physical constants (e.g., 'speed of light') without numerical substitution.
- Include a space between compound units (e.g., "Ω m" for "ohm*meter").
- To solve for a variable in an equation with units, consider solving a corresponding equation without units; exclude counting units (e.g., books), include genuine units (e.g., kg).
- If data for multiple properties is needed, make separate calls for each property.
- If a WolframAlpha result is not relevant to the query:
 -- If Wolfram provides multiple 'Assumptions' for a query, choose the more relevant one(s) without explaining the initial result. If you are unsure, ask the user to choose.
 -- Re-send the exact same 'input' with NO modifications, and add the 'assumption' parameter, formatted as a list, with the relevant values.
 -- ONLY simplify or rephrase the initial query if a more relevant 'Assumption' or other input suggestions are not provided.
 -- Do not explain each step unless user input is needed. Proceed directly to making a better API call based on the available assumptions.
'''



# https://products.wolframalpha.com/api/documentation?scrollTo=using-assumptions


# wa has a lot of image based responses, hard to use now

@ProxyRegistrator(
    path='wa',
    name='Wolfram Alpha API',
    description=(
        "Query the Wolfram Alpha. WolframAlpha is an answer engine developed by Wolfram Research. It is offered as an online service that answers factual queries by computing answers from externally sourced data."
    )
)
class WAProxy(BaseProxy):
    def __init__(self, cutoff_date: str = None, cache: bool = True, **kwargs):
        super().__init__(cutoff_date=cutoff_date, use_cache=cache, **kwargs)
        self.api_key = os.getenv("WA_API_DEV")
        self.api_key_name = "appid"
        self.base_url = "https://www.wolframalpha.com/api/v1"
        self.additional_docs = {
            'Use Assumptions': USE_ASSUMPTIONS
        }

    def _call_api(self, url: str, params: dict, endpoint_info: dict, headers: dict) -> dict:
        """
        Helper method to call the API using the requests library.
        """
        response = U.call_api(url, params, headers, self.use_cache, json_response=False)
        return {'response': response.text}

    @BaseProxy.endpoint(
        category='Query',
        endpoint='llm-api',
        name='Wolfram Alpha LLM API',
        description='An LLM-based Wolfram Alpha service.',
        params={
            "input*": (str, "10 densest elemental metals"),
            'assumption': (str, None),
            # 'units': (str, "metric"),
        },
        response={
            'response': '''
                Query:
                "10 densest elemental metals"

                Input interpretation:
                10 densest metallic elements | by mass density

                Periodic table location:
                image: https://www6b3.wolframalpha.com/Calculate/MSP/MSP1006242c78id33f7hia300002b6b6feg7gd456e1?MSPStoreType=image/png&s=3

                Thermodynamic properties:
                phase at STP | all | solid
                (properties at standard conditions)

                Wolfram|Alpha website result for "10 densest elemental metals":
                https://www.wolframalpha.com/input?i=10+densest+elemental+metals
            '''
        }
    )
    def query(self, params: dict) -> dict: 
        '''
        Query Wolfram Alpha API
        
        input (required):
            - Function: The input to the query
            - Sample values: "10 densest elemental metals"

        assumption (optional):
            - Function: Specifies an assumption, such as the meaning of a word or the value of a formula variable	
            - Sample values: "*C.pi-_*Movie", "DateOrder_**Day.Month.Year--"	
            - Default values: Assumptions made implicitly by the API	
            - Notes: Values for this parameter are given by the input properties of <value> subelements of <assumption> elements in XML results
            - See: the documentation for the # Use Assumptions section
        '''
        return params
