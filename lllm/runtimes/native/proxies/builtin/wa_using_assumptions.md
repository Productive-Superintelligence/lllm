# Using Assumptions

Wolfram|Alpha makes numerous assumptions when analyzing a query and deciding how to present its results. A simple example is a word that can refer to multiple things, like "pi", which is a well-known mathematical constant but is also the name of a movie. Other classes of assumptions are the meaning of a unit abbreviation like "m", which could be meters or minutes, or the default value of a variable in a formula, or whether 12/13/2001 is a date or a computation. Using the API, you can programmatically invoke these assumption values to alter the output of a query.


## Listing of Assumption Types

This is a complete listing of possible assumptions, categorized by functionality. Each assumption name links to a more in-depth explanation of the parameter, its usage and available options.

### Categorical assumptions

 — Clash: choose among possible interpretation categories for a query (e.g. "pi" could be a named constant or a movie)
 — MultiClash: choose among possible interpretation categories for in Structure of Assumptions dividual words in a query (e.g. in "log 0.5", "log" could be a PDF or a plotting function)
 — SubCategory: choose among similar results within the same interpretation category (e.g. "hamburger" could be a basic hamburger, McDonald's hamburger, etc.)
 — Attribute: choose specific traits of results within a SubCategory (e.g. for a "basic hamburger" you can specify patty size, condiments, etc.)

### Mathematical assumptions
 — Unit: choose among possible interpretations of a query as a unit of measure (e.g. "m" could be meters or minutes)
 — AngleUnit: choose whether to interpret angles in degrees or radians (e.g. "sin(30)" could be 30 degrees or 30 radians)
 — Function: choose among possible interpretations of a query as a mathematical function (e.g. "log" could be the natural logarithm or the base 10 logarithm)
 — ListOrTimes: choose whether to interpret space-separated elements as a list of items or as multiplication (e.g. "3 x" could be {3, x} or 3×x)
 — ListOrNumber: choose whether to treat commas between numbers as thousands separators or list delimiters (e.g. "1,234.5" could be 1234.5 or {1, 234.5})
 — CoordinateSystem: choose which coordinate system to use, given an ambiguous query (e.g. "div(x rho, y z, z x)" could refer to Cartesian3D or Cylindrical3D)
 — I: choose whether to interpret "i" as a variable or as the imaginary unit (e.g. "5x+3i" could be a complex number with one variable or an expression with two variables)
 — NumberBase: choose which number base to use, given an ambiguous query (e.g. "100110101" could be binary or decimal)
 — MixedFraction: choose whether to interpret input as a mixed fraction or as multiplication (e.g. "3 1/2" could be 3½ or 3×½)

### Date and time assumptions
 — TimeAMOrPM: choose whether a time is in the morning or the afternoon (e.g. "3:00" could be 3am or 3pm)
 — DateOrder: choose the order of day, year and month elements in a date (e.g. "12/11/1996" could be Dec. 11, 1996, or Nov. 12, 1996)
 — MortalityYearBOB: choose whether a mortality rate refers to a data year or a birth year (e.g. "life expectancy France 1910" could refer to statistics from 1910 or those for people born in 1910)

### Scientific assumptions
 — TideStation: choose among tide stations in a tide query (e.g. "tides Seattle" could use Seattle, Bangor or Tacoma tide stations as a source)

### Formula input assumptions
 — FormulaSelect: choose among possible interpretations of a query as a mathematical formula (e.g. "Doppler shift" can refer to either the classical or relativistic Doppler shift)
 — FormulaSolve: choose which value to solve for in a formula (e.g. solving for sound speed in the Doppler shift formula)
 — FormulaVariable: supply a non-default value to a specific variable in a formula (e.g. give the "Doppler shift" query a source speed of 30 m/s)
 — FormulaVariableOption: choose among different representations of a variable or variables in a formula (e.g. the Doppler shift formula can split the frequency reduction factor into source and observed frequencies)
 — FormulaVariableInclude: choose extra variables to include in a formula (e.g. "Doppler shift" can also include the observer's speed and wind speed)

## Structure of Assumptions

On the website, the space just above the first pod is used to describe assumptions used and give the user the option of changing them. Clicking any of the links in this area will prompt Wolfram|Alpha to send a new query based on the given information. Here is the assumptions area from the result for "pi":

*Assuming "pi" is a mathematical constant | Use a character or referring to a mathematical definition or a class of mathematical terms or a movie or a word instead*

The API makes this same information available via the <assumptions> element, which, if present, is a top-level subelement of <queryresult>. Each assumption is represented as a separate <assumption> subelement containing a sequence of <value> subelements, one for each possible value of the assumption. The <assumption> element has a type attribute that tells you what class of assumption it is. In this example, it is a "Clash" assumption, which is generated when a word in the query can refer to multiple different entities. Here is the <assumptions> element in the "pi" query:

<assumptions count="1">
  <assumption type="Clash" word="pi" template="Assuming "${word}" is ${desc1}. Use as ${desc2} instead" count="6">
    <value name="NamedConstant" desc="a mathematical constant" input="*C.pi-_*NamedConstant-"/>
    <value name="Character" desc="a character" input="*C.pi-_*Character-"/>
    <value name="MathWorld" desc=" referring to a mathematical definition" input="*C.pi-_*MathWorld-"/>
    <value name="MathWorldClass" desc="a class of mathematical terms" input="*C.pi-_*MathWorldClass-"/>
    <value name="Movie" desc="a movie" input="*C.pi-_*Movie-"/>
    <value name="Word" desc="a word" input="*C.pi-_*Word-"/>
  </assumption>
</assumptions>

Each <value> element has three attributes: name, which is a unique internal identifier (often with some descriptive value to the programmer); desc, which is a textual description suitable for displaying to users; and input, which gives the exact parameter value needed to invoke this assumption in a subsequent query. The template attribute shows the natural language statement normally displayed by the Wolfram|Alpha website to allow users to alter the current query. In this example, ${word} refers to the input, "pi", ${desc1} refers to the current interpretation of "a mathematical constant" and ${desc2} refers to a new interpretation (e.g. "a character" or "a movie"). These values are populated from information within the XML, with each ${wordX} drawing from the "word" attribute of the <assumption> element and each ${descX} drawing from the "desc" attributes of individual <value> elements. The first-listed <value> element always names the assumption value that was in effect for the current query.

Many other assumption types exist. For instance, Wolfram|Alpha interprets the query "12/5/1999" as a date; however, it's ambiguous whether this is written in the order month/day/year or day/month/year, so an assumption is generated. Here is the relevant <assumption> element, which has type "DateOrder". The month/day/year value is listed first, which means that it is the value that was used to get the current result:

<assumption type="DateOrder" template="Assuming ${desc1}. Use ${desc2} instead" count="2">
  <value name="MonthDayYear" desc="month/day/year" input="DateOrder_**Month.Day.Year--"/>
  <value name="DayMonthYear" desc="day/month/year" input="DateOrder_**Day.Month.Year--"/>
</assumption>

The next section includes a listing of the main assumption types that can be generated, with examples. Note that formula input assumptions are covered in the "Formulas with Input Fields" section.

Applying Assumptions
To apply an assumption in a query, use the assumption parameter. The value you pass for this parameter is the same as the string found in the input attribute of a <value> element returned from a previous query. Here is how to invoke the query "pi", but specify that you want pi treated as the name of a movie. The assumption string here was taken from the earlier <assumptions> output for this query: input=pi&assumption=*C.pi-_*Movie-

Here is how you would modify the "12/5/1999" query to change the date order to day/month/year: input=12%2F5%2F1999&assumption=DateOrder_**Day.Month.Year--

The values for the assumption parameter are complicated-looking strings, but you don't have to understand their syntax—just use the tokens that you are given in the <assumptions> output of a previous query. You can apply more than one assumption in a given query by including multiple "assumption=value" specifications in the params.

#### Clash

The Clash assumption is generated when a word can represent different categories of things, such as "pi" being treated as a mathematical constant, a movie, a character or simply as a word. Here is a typical example of a Clash <assumption> element, generated from the query "pi":

<assumptions count="1">
  <assumption type="Clash" word="pi" template="Assuming "${word}" is ${desc1}. Use as ${desc2} instead" count="6">
    <value name="NamedConstant" desc="a mathematical constant" input="*C.pi-_*NamedConstant-"/>
    <value name="Character" desc="a character" input="*C.pi-_*Character-"/>
    <value name="MathWorld" desc=" referring to a mathematical definition" input="*C.pi-_*MathWorld-"/>
    <value name="MathWorldClass" desc="a class of mathematical terms" input="*C.pi-_*MathWorldClass-"/>
    <value name="Movie" desc="a movie" input="*C.pi-_*Movie-"/>
    <value name="Word" desc="a word" input="*C.pi-_*Word-"/>
  </assumption>
</assumptions>

#### MultiClash

The MultiClash assumption is a type of clash where multiple overlapping strings can have different interpretations. An example is the query "delta sigma", where the whole phrase can be interpreted as a formula, or the word "delta" can be interpreted as a financial entity, a variable or an administrative division:

<assumptions count="1">
  <assumption type="MultiClash" word="" template="Assuming ${word1} is referring to ${desc1}. Use "${word2}" as ${desc2}. Use "${word3}" as ${desc3}. Use "${word4}" as ${desc4}." count="4">
    <value name="Financial" word="delta" desc="a financial entity" input="*MC.%7E-_*Financial-"/>
    <value name="Variable" word="delta" desc="a variable" input="*MC.%7E-_*Variable-"/>
    <value name="AdministrativeDivision" word="delta" desc="an administrative division" input="*MC.%7E-_*AdministrativeDivision-"/>
    <value name="Formula" word="" desc="a formula" input="*MC.%7E-_*Formula-"/>
  </assumption>
</assumptions>

In contrast to Clash, the MultiClash assumption can interpret phrases in addition to individual words.

#### SubCategory

The SubCategory assumption is similar to the Clash type in that a word can refer to multiple types of entities, but for SubCategory all the interpretations are within the same overall category. An example is the query "hamburger", which generates a SubCategory assumption for different types of hamburger (basic hamburger, McDonald's hamburger, Burger King hamburger, etc.) The hamburger query also generates a Clash assumption over whether hamburger should be treated as a type of food or a simple word, but given that Wolfram|Alpha is treating hamburger as a type of food in this query, it also can be resolved into subcategories of hamburger. Here is the SubCategory <assumption> element from the "hamburger" query:

<assumption type="SubCategory" word="hamburger" template="Assuming ${desc1}. Use ${desc2} instead" count="8">
  <value name="Hamburger" desc="hamburger" input="*DPClash.ExpandedFoodE.hamburger-_*Hamburger-"/>
  <value name="GroundBeefPatty" desc="ground beef patty" input="*DPClash.ExpandedFoodE.hamburger-_*GroundBeefPatty-"/>
  <value name="GroundBeef" desc="ground beef" input="*DPClash.ExpandedFoodE.hamburger-_*GroundBeef-"/>
   ... Remaining elements deleted for brevity ...
</assumption>

#### Attribute

You can think of the Attribute assumption as the next step down in the sequence of Clash and SubCategory. Wolfram|Alpha emits an Attribute assumption to allow you to modify an attribute of an already well-characterized entity. In the query "hamburger", Wolfram|Alpha assumes you mean that hamburger is a food item (although it gives you a Clash assumption to modify this) and that you mean a "basic" hamburger (and it gives you a SubCategory assumption to make this, say, a McDonald's hamburger). It also gives you an Attribute assumption to modify details like patty size and whether it has condiments. Here is the Attribute <assumption> element from the "hamburger" query:

<assumption type="Attribute" word="Hamburger" template="Assuming ${desc1}. Use ${desc2} instead" count="9">
  <value name="Hamburger"
         desc="any type of hamburger"
         input="*EAC.ExpandedFood.Hamburger-_**a.Hamburger--"/>
  <value name="{Food:FoodSize -> Food:LargePatty, Food:PattyCount -> Food:Single, Food:Variety -> Food:FastFood, Food:Variety -> Food:Plain}"
         desc="hamburger, fast food, large patty, plain, single"
         input="*EAC.ExpandedFood.Hamburger-_**Hamburger.*Food%3AFoodSize_Food%3ALargePatty.Food%3APattyCount_Food%3ASingle.Food%3AVariety_Food%3AFastFood.Food%3AVariety_Food%3APlain---"/>
  ... Remaining elements deleted for brevity ...
</assumption>

The names and input values for Attribute assumptions can become rather cryptic, but the desc attributes are always given in natural language for clarity.

#### Unit

The Unit assumption is generated when a word is interpreted as a unit abbreviation, but it is ambiguous as to what unit it represents. An example is "m", meaning either meters or minutes. Here is a typical example of a Unit <assumption> element, this generated from the query "10 m":

<assumption type="Unit" word="m" template="Assuming ${desc1} for "${word}". Use ${desc2} instead" count="3">
  <value name="Meters" desc="meters" input="UnitClash_*m.*Meters--"/>
  <value name="MinimsUS" desc="US minims of volume" input="UnitClash_*m.*MinimsUS--"/>
  <value name="Minutes" desc="minutes of time" input="UnitClash_*m.*Minutes--"/>
</assumption>

#### AngleUnit

The AngleUnit assumption is generated when a number is interpreted as a unit of angle, but it is ambiguous whether it should be interpreted as degrees or radians. This assumption type always has two <value> elements, one for the assumption of degrees and the other for the assumption of radians. Here is the AngleUnit <assumption> element generated from the query "sin(30)":

<assumption type="AngleUnit" template="Assuming trigonometric arguments in ${desc1}. Use ${desc2} instead" count="2">
  <value name="D" desc="degrees" input="TrigRD_D"/>
  <value name="R" desc="radians" input="TrigRD_R"/>
</assumption>

#### Function

The Function assumption is generated when a word is interpreted as referring to a mathematical function, but it is ambiguous which function is meant. An example is "log", meaning either log base e or log base 10. Here is a typical example of a Function <assumption> element, this generated from the query "log 20":

<assumption type="Function" word="log" template="Assuming "${word}" is ${desc1}. Use ${desc2} instead" count="2">
  <value name="Log" desc="the natural logarithm" input="*FunClash.log-_*Log.Log10-"/>
  <value name="Log10" desc="the base 10 logarithm" input="*FunClash.log-_*Log10.Log-"/>
</assumption>

#### TimeAMOrPM

When Wolfram|Alpha recognizes a string in a query as referring to a time, and it is ambiguous as to whether it represents AM or PM, a TimeAMOrPM assumption is generated. Here is the <assumption> element from the query "3:00":

<assumption type="TimeAMOrPM" template="Assuming ${desc1}. Use ${desc2} instead" count="2">
  <value name="pm" desc="3:00 PM" input="*TimeAMOrPM.*%7FAutomatic.%7FAutomatic.%7FAutomatic.%7F3.%7F0.%7FAutomatic--_pm"/>
  <value name="am" desc="AM" input="*TimeAMOrPM.*%7FAutomatic.%7FAutomatic.%7FAutomatic.%7F3.%7F0.%7FAutomatic--_am"/>
</assumption>

There are always two <value> elements in this assumption: one for am and one for pm. As always, the first-listed one is the current value for the assumption, and this will depend on what time of day the query is executed.

#### DateOrder

When Wolfram|Alpha recognizes a string in a query as referring to a date in numerical format, and it is ambiguous as to the order of the day, month and year elements (such as 12/11/1996), a <assumption> assumption is generated. Here is the <assumption> element from the query "12/11/1996":

<assumption type="DateOrder" template="Assuming ${desc1}. Use ${desc2} instead" count="2">
  <value name="MonthDayYear" desc="month/day/year" input="DateOrder_**Month.Day.Year--"/>
  <value name="DayMonthYear" desc="day/month/year" input="DateOrder_**Day.Month.Year--"/>
</assumption>

The number and order of <value> elements depends on specifics of the date string in the query, and also on the locale of the caller. The name attributes will be a combination of day, month and year in the corresponding order.

#### MortalityYearDOB

The MortalityYearDOB assumption is a very specialized type generated in some mortality-related queries, such as "life expectancy France 1910". The year 1910 could refer to the year of the data (that is, life expectancy data from France in the year 1910), or the year of birth ("life expectancy data in France for people born in 1910"). The MortalityYearDOB assumption distinguishes between those two meanings. Here is the <assumption> element from that query:

<assumption type="MortalityYearDOB" word="1910" template="Assuming ${word} is ${desc1}. Use as ${desc2} instead" count="2">
  <value name="Year" desc="the year of the data" input="MortYrDOB_*Yr.1910-"/>
  <value name="DateOfBirth" desc="the year of birth" input="MortYrDOB_*DOB.1910-"/>
</assumption>

The MortalityYearDOB assumption always has two <value> elements: one named "Year" and one named "DateOfBirth."

#### ListOrTimes

The ListOrTimes assumption is generated when a query contains elements separated by spaces and it is unclear whether this is to be interpreted as multiplication or a list of values. For example, the query "3 x" is interpreted as 3×x, but it could also be the list {3, x}. Here is the ListOrTimes <assumption> element from that query:

<assumption type="ListOrTimes" template="Assuming ${desc1}. Use ${desc2} instead" count="2">
  <value name="Times" desc="multiplication" input="ListOrTimes_Times"/>
  <value name="List" desc="a list" input="ListOrTimes_List"/>
</assumption>

The ListOrTimes assumption always has two <value> elements: one named "List" and one named "Times." There is no word attribute in the <assumption> element for this type.

#### ListOrNumber

The ListOrNumber assumption is generated when a query contains a string that could be either a number with a comma as a thousands separator or a list of two separate numbers, such as the query "1,234.5." Here is the ListOrNumber <assumption> element from that query:

<assumption type="ListOrNumber" word="1,234.5" template="Assuming ${word} is a ${desc1}. Use ${word} as a ${desc2} instead" count="2">
  <value name="Number" desc="number" input="ListOrNumber_*Number.1%2C234%21.5-"/>
  <value name="List" desc="list" input="ListOrNumber_*List.1%2C234%21.5-"/>
</assumption>

The ListOrNumber assumption always has two <value> elements: one named "List" and one named "Number."

#### CoordinateSystem

The CoordinateSystem assumption is generated when it is ambiguous which coordinate system a query refers to. For example, the query "div(x rho,y z,z x)" mixes elements from standard notation for 3D Cartesian coordinates and cylindrical coordinates. Here is the CoordinateSystem <assumption> element from that query:

<assumption type="CoordinateSystem" template="Using ${desc1}. Use ${desc2} instead" count="2">
  <value name="Cartesian3D" desc="3D Cartesian coordinates" input="CoordinateSystem_*Cartesian3D-"/>
  <value name="Cylindrical3D" desc="cylindrical coordinates" input="CoordinateSystem_*Cylindrical3D-"/>
</assumption>

The possible values for the name attribute are Cartesian2D, Cartesian3D, Polar2D, Cylindrical3D, Spherical3D, General2D and General3D. There is no word attribute in the <assumption> element for this type.

#### I

The I assumption is generated when a query uses "i" in a way that could refer to a simple variable name (similar to, say, "x") or the mathematical constant equal to the square root of -1. Here is what this assumption looks like:

<assumption type="I" template="Assuming ${desc1}. Use ${desc2} instead" count="2">
  <value name="ImaginaryI" desc="i is the imaginary unit" input="i_ImaginaryI"/>
  <value name="Variable" desc="i is a variable" input="i_Variable"/>
</assumption>

The I assumption always has two <value> elements: one named "ImaginaryI" and one named "Variable". There is no word attribute in the <assumption> element for this type, as it always refers to the letter "i".

#### NumberBase

The NumberBase assumption is generated when a number could be interpreted as being written in more than one base, such as "1011001110", which looks like a binary number (base 2) but could also be base 10 (it could be other bases as well, but those are rarely used and thus do not occur as assumption values). Here is the NumberBase <assumption> element from that query:

<assumption type="NumberBase" word="1011001110" template="Assuming ${word} is ${desc1}. Use ${desc2} instead" count="2">
  <value name="Binary" desc="binary" input="NumberBase_*Binary.1011001110-"/>
  <value name="Decimal" desc="decimal" input="NumberBase_*Decimal.1011001110-"/>
</assumption>

At the present time, the only possible <value> elements for this assumption are "Decimal" and "Binary".

#### MixedFraction

The MixedFraction assumption is generated when a string could be interpreted as either a mixed fraction or as multiplication, such as "3 1/2". Here is the MixedFraction <assumption> element from that query:

<assumption type="MixedFraction" word="3 1/2" template="Assuming ${word} is a ${desc1}. Use ${word} as a ${desc2} instead" count="2">
  <value name="Mix" desc="mixed fraction" input="MixFrac_*Mix.3+1%2F2-"/>
  <value name="Mult" desc="product" input="MixFrac_*Mult.3+1%2F2-"/>
</assumption>

The MixedFraction assumption always has two <value> elements: one named "Mix" and one named "Mult".

#### TideStation

The TideStation assumption is generated in tide-related queries. It distinguishes between different tide stations. Here is an example from the "tides Seattle" query:

<assumption type="TideStation" template="Using ${desc1}. Use ${desc2} instead" count="5">
  <value name="PrimaryStation" desc="nearest primary station" input="TideStation_PrimaryStation"/>
  <value name="NearestStation" desc="nearest station" input="TideStation_NearestStation"/>
  <value name="Seattle, Washington (1.4 mi)" desc="Seattle, Washington (1.4 mi)" input="TideStation_*UnitedStates.9447130.PrimaryStation-"/>
  <value name="Bangor, Washington (19.6 mi)" desc="Bangor, Washington (19.6 mi)" input="TideStation_*UnitedStates.9445133.PrimaryStation-"/>
  <value name="Tacoma, Washington (24.6 mi)" desc="Tacoma, Washington (24.6 mi)" input="TideStation_*UnitedStates.9446484.PrimaryStation-"/>
</assumption>

Note that the default station for these queries is the nearest station to the caller's location.

### Formulas with Input Fields

Some Wolfram|Alpha inputs are interpreted as referring to mathematical formulas. In such cases, the Wolfram|Alpha website provides a user interface for controlling aspects of the formula, such as what variable is being solved for and what values the variables should be given. Formula solving is part of the Wolfram|Alpha Assumptions facility, and the API gives you complete control over all aspects of formula manipulation.

Here is the <assumption> section from the API output for the "Doppler shift" query:

<assumptions count="7">
  <assumption type="Clash" word="Doppler shift" template="Assuming "${word}" is ${desc1}. Use as ${desc2} instead" count="3">...</assumption>
  <assumption type="FormulaSolve" template="Calculate ${desc1}" count="3">
    <value name="DopplerShift.DopplerRatio" desc="frequency reduction factor" input="*FS-_**DopplerShift.DopplerRatio--"/>
    <value name="DopplerShift.vs" desc="speed of the source away from the observer" input="*FS-_**DopplerShift.vs--"/>
    <value name="DopplerShift.c" desc="sound speed" input="*FS-_**DopplerShift.c--"/>
  </assumption>
  <assumption type="FormulaSelect" template="Assuming ${desc1}. Use ${desc2} instead" count="2">
    <value name="DopplerShift" desc="Doppler shift" input="FSelect_*DopplerShift-"/>
    <value name="RelativisticDopplerShift" desc="relativistic Doppler shift" input="FSelect_**RelativisticDopplerShift--"/>
  </assumption>
  <assumption type="FormulaVariable" desc="speed of the source away from the observer" current="1" count="1">
    <value name="DopplerShift.vs" desc="10 m/s" valid="true" input="*F.DopplerShift.vs-_10+m%2Fs"/>
  </assumption>
  <assumption type="FormulaVariable" desc="sound speed" current="1" count="1">
    <value name="DopplerShift.c" desc="340.3 m/s" valid="true" input="*F.DopplerShift.c-_340.3+m%2Fs"/>
  </assumption>
  <assumption type="FormulaVariableOption" template="Assuming ${desc1}. Use ${desc2} instead" count="2">
    <value name="DopplerShift.DopplerRatio" desc="frequency reduction factor" input="*FVarOpt-_**DopplerShift.DopplerRatio--"/>
    <value name="DopplerShift.fo,DopplerShift.fs" desc="frequency observed and frequency at the source" input="*FVarOpt-_**DopplerShift.fo-.*DopplerShift.fs--"/>
  </assumption>
  <assumption type="FormulaVariableInclude" template="Also include ${desc1}" count="2">
    <value name="DopplerShift.vo" desc="speed of the observer" input="*FVarOpt-_**DopplerShift.vo-.*DopplerShift.DopplerRatio--"/>
    <value name="DopplerShift.vw" desc="wind speed" input="*FVarOpt-_**DopplerShift.vw-.*DopplerShift.DopplerRatio--"/>
  </assumption>
</assumptions>

Most of this output should be self-explanatory in the context of the earlier discussion of assumptions in general. Using the information in this output, you could build a user interface that allows your users to interact with the Doppler formula in exactly the same way as the Wolfram|Alpha website.

### Applying Formula Assumptions

You apply formula-related assumptions just like other types of assumptions—by using the value of an <assumption> element's input attribute in a subsequent query. For example, to perform the Doppler shift query with wind speed as an added variable, you would use this params (written in url form): input=Doppler+shift&assumption=*FVarOpt-_**DopplerShift.vw-.*DopplerShift.DopplerRatio--
You can specify as many assumptions as you want in a single query, which is often necessary when working with formulas (such as setting the value of multiple variables). To do this, include multiple assumption=value specifications in the params.

### Formula Assumption Types

Here is a guide to the five different formula assumption types, note that not all formulas will have all of these assumptions available:

#### FormulaSelect

Some queries have more than one formula that applies. The FormulaSelect assumption allows you to choose the one you want. In this Doppler example, you can choose the classical Doppler shift formula (the default) or the relativistic one:

<assumption type="FormulaSelect" template="Assuming ${desc1}. Use ${desc2} instead" count="2">
  <value name="DopplerShift" desc="Doppler shift" input="FSelect_*DopplerShift-"/>
  <value name="RelativisticDopplerShift" desc="relativistic Doppler shift" input="FSelect_**RelativisticDopplerShift--"/>
</assumption>

#### FormulaSolve

Formulas can be rearranged to solve for different variables. The FormulaSolve assumption lets you pick which one you want. In this example, the variables are the frequency reduction factor (f0/fn, treated as a single entity), the speed of sound (c) and the speed of the source (vs). Notice in the Result pod it shows a value for frequency reduction factor, which is the current choice for the variable to solve for. If you were to choose a different variable to solve for, it would show that value in this pod:

<assumption type="FormulaSolve" template="Calculate ${desc1}" count="3">
  <value name="DopplerShift.DopplerRatio" desc="frequency reduction factor" input="*FS-_**DopplerShift.DopplerRatio--"/>
  <value name="DopplerShift.vs" desc="speed of the source away from the observer" input="*FS-_**DopplerShift.vs--"/>
  <value name="DopplerShift.c" desc="sound speed" input="*FS-_**DopplerShift.c--"/>
</assumption>

#### FormulaVariable

The FormulaVariable assumption lets you supply a value for a variable in a formula. It corresponds to an input field or pull-down menu of choices on the website. You have to understand a few extra details when applying a FormulaVariable assumption. Let's take a closer look at the <assumption> element from the "Doppler shift" query that deals with the value of the speed of the source:

<assumption type="FormulaVariable" desc="speed of the source away from the observer" current="1" count="1">
  <value name="DopplerShift.vs"
         desc="10 m/s"
         valid="true"
         input="*F.DopplerShift.vs-_10+m%2Fs"/>
</assumption>

When you see such an assumption in output, you might choose to provide your users with an input field or other means to specify a value. The label for this input field would be the desc attribute of the <assumption> element. The count attribute gives the number of <value> elements. For variables that take an arbitrary value, typically entered via an input field, the count will always be 1, but for variables that take one of a fixed set of values, typically represented as a pull-down menu of choices, the count will be the number of possible choices, with one <value> element for each possibility.

For the moment, we restrict our attention to the common case of a variable that can take any user-specified value. The single <value> element for this assumption will have a desc attribute that gives the currently assumed value for the variable. If you were providing an input field for your users to specify a new value, you would probably want to "prime" the input field with this initial value, like the Wolfram|Alpha website does. To specify a different value, you need to work with the value of the input attribute. You can ignore everything that comes before the "-_" in the string. What comes after is the value—in this case, "10+m%2Fs", the URL-encoded form of "10 m/s". To specify a different value for this variable, replace what comes after the "-_" character pair with the URL-encoded new value. Here is how to set the speed of the source to 6.5 m/s:
input=Doppler+shift&assumption=*F.DopplerShift.vs-_6.5+m%2Fs

Wolfram|Alpha intelligently parses the value, so it understands that "m/s" means "meters per second." Those are the default units for this value, so you could leave the units specification out entirely and just give a number. You could also write out "meters/second", "meters per second", etc. If you submit a value that cannot be understood by Wolfram|Alpha, then the corresponding <value> element in the result will have the valid=false attribute. For example, if you try to set the value to 6.5 f/s, then this is what the resulting <assumption> element looks like:

<assumption type="FormulaVariable" desc="speed of the source away from the observer" current="1" count="1">
  <value name="DopplerShift.vs"
         desc="6.5 f/s"
         valid="false"
         input="*F.DopplerShift.vs-_6.5+f%2Fs"/>
</assumption>

The problem is that Wolfram|Alpha does not understand the "f/s" units. To specify feet per second, you would need to use "ft/s", "feet/second", etc.

Now consider the case where a formula variable assumption can take on only a fixed set of values. These are not common in Wolfram|Alpha, but an example query that generates such assumptions is "heart disease", which on the website produces a fillable form with several pull-down menus allowing you to choose whether the person in question is a smoker, diabetic, etc.:

gender: male
age: 36 yr
LDL cholesterol: 110 mg/dL
HDL cholesterol: 54 mg/dL
systolic blood pressure: 120 mmHg
diastolic blood pressure: 80 mmHg
smoker: no
diabetic: no

Note that the fillable fields in this formula work as described in the previous example. However, the pull-down menus are derived from assumptions with only a few specific options available. Here is the XML returned by the API for one of these assumptions:

<assumption type="FormulaVariable" desc="gender" current="1" count="2">
  <value name="Gender:Male"
         desc="male"
         valid="true"
         input="*FP.HeartDisease.gender-_Gender%3AMale"/>
  <value name="Gender:Female"
         desc="female"
         valid="true"
         input="*FP.HeartDisease.gender-_Gender%3AFemale"/>
</assumption>

In all other assumption types, the first-listed <value> element names the currently active assumption, but in this one case that rule is violated. Instead, the current attribute of <assumption> gives the index of the <value> element that is currently active (in this case, "1" for "male" and "2" for "female"). In this way, the natural order of the different values is preserved, without artificially moving the current value to the top of the list. Whether representing these options in a pull-down menu, as radio buttons, or using some other representation, it will probably make sense to preserve the order of the <value> elements. The process for applying a fixed FormulaVariable assumption is the same as for other types described in the "Using Assumptions" section.

#### FormulaVariableInclude

The FormulaVariableInclude assumption lets you add additional variables into a formula. For simplicity, Wolfram|Alpha presents the Doppler shift formula with a small number of variables, but it knows how to include two more: the speed of the observer and the wind speed. On the website, if you click to add one of these variables, the formula will change to include this variable, the tabular results will get an extra row for it and you will get an extra input field to enter its value.

<assumption type="FormulaVariableInclude" template="Also include ${desc1}" count="2">
  <value name="DopplerShift.vo" desc="speed of the observer" input="*FVarOpt-_**DopplerShift.vo-.*DopplerShift.DopplerRatio--"/>
  <value name="DopplerShift.vw" desc="wind speed" input="*FVarOpt-_**DopplerShift.vw-.*DopplerShift.DopplerRatio--"/>
</assumption>

#### FormulaVariableOption

Wolfram|Alpha can sometimes present the same basic formula in terms of a different set of variables. In the Doppler example, you can choose to have the frequency reduction factor (f0/fn) broken up into two separate variables (f0 and fn). You're not substituting a completely different formula (like FormulaSelect) or simply adding a new variable (like FormulaVariableInclude).

<assumption type="FormulaVariableOption" template="Assuming ${desc1}. Use ${desc2} instead" count="2">
  <value name="DopplerShift.DopplerRatio" desc="frequency reduction factor" input="*FVarOpt-_**DopplerShift.DopplerRatio--"/>
  <value name="DopplerShift.fo,DopplerShift.fs" desc="frequency observed and frequency at the source" input="*FVarOpt-_**DopplerShift.fo-.*DopplerShift.fs--"/>
</assumption>
