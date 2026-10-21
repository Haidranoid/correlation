"""Station list for the data-weather agent (imported by the parsers)."""

# key, GHCN id, display name, role, snow?, wind?, degree days?
# dd: HDD/CDD (base 65F) only at market and energy centres; CPC regional DD cover the rest.
# snow: SNOW kept only where snowfall is common enough to matter.
# wind: AWND kept for a subset (wind-power regions, coastal storm exposure, big centres).
STATIONS = [
    # population and market centres
    ("nyc", "USW00094728", "New York Central Park", "market centre", True, True, True),
    ("chi", "USW00094846", "Chicago O'Hare", "market centre", True, True, True),
    ("bos", "USW00014739", "Boston Logan", "market centre", True, True, True),
    ("phl", "USW00013739", "Philadelphia Intl", "market centre", True, False, True),
    ("dca", "USW00013743", "Washington Reagan National", "market centre", True, False, True),
    ("atl", "USW00013874", "Atlanta Hartsfield-Jackson", "market centre", False, False, True),
    ("dfw", "USW00003927", "Dallas-Fort Worth", "market centre", False, True, True),
    ("hou", "USW00012960", "Houston Intercontinental", "market/energy centre", False, True, True),
    ("lax", "USW00023174", "Los Angeles Intl", "market centre", False, False, True),
    ("sfo", "USW00023234", "San Francisco Intl", "market centre", False, False, True),
    ("sea", "USW00024233", "Seattle-Tacoma", "market centre", False, False, True),
    ("den", "USW00003017", "Denver Intl", "market centre", True, False, True),
    ("phx", "USW00023183", "Phoenix Sky Harbor", "market centre", False, False, True),
    ("mia", "USW00012839", "Miami Intl", "market centre", False, True, True),
    ("msp", "USW00014922", "Minneapolis-St Paul", "market centre", True, False, True),
    ("dtw", "USW00094847", "Detroit Metro", "market centre", True, False, True),
    # agriculture
    ("dsm", "USW00014933", "Des Moines", "agriculture (corn/soy)", True, True, False),
    ("pia", "USW00014842", "Peoria IL", "agriculture (corn/soy)", True, False, False),
    ("oma", "USW00014942", "Omaha Eppley", "agriculture (corn/cattle)", True, False, False),
    ("mci", "USW00003947", "Kansas City Intl", "agriculture (wheat market)", True, False, False),
    ("ict", "USW00003928", "Wichita", "agriculture (HRW wheat)", True, True, False),
    ("far", "USW00014914", "Fargo Hector", "agriculture (spring wheat/soy)", True, True, False),
    ("fat", "USW00093193", "Fresno Yosemite", "agriculture (Central Valley)", False, False, False),
    ("mem", "USW00013893", "Memphis", "agriculture (cotton/soy, river)", False, False, False),
    ("lbb", "USW00023042", "Lubbock", "agriculture (cotton) / wind", False, True, False),
    # energy
    ("lch", "USW00003937", "Lake Charles LA", "energy (Henry Hub/LNG/refining)", False, True, True),
    ("msy", "USW00012916", "New Orleans", "energy (Gulf refining/ports)", False, False, True),
    ("okc", "USW00013967", "Oklahoma City", "energy (Cushing area) / wind", False, True, True),
    ("maf", "USW00023023", "Midland-Odessa TX", "energy (Permian)", False, True, True),
    ("pit", "USW00094823", "Pittsburgh", "energy (Appalachian gas)", True, False, True),
    # foreign financial centres
    ("lon", "UKE00107650", "London Heathrow", "foreign centre", False, False, False),
    ("syd", "ASN00066037", "Sydney Airport", "foreign centre (Asia-Pacific; replaces Tokyo)", False, False, False),
    ("fra", "GME00122362", "Frankfurt/Main-Westend", "foreign centre", False, False, False),
]
