#!/usr/bin/env python3

from collections import OrderedDict, namedtuple
import json
import logging
from datetime import datetime
import re
import pandas as pd

COLORS = ('Black', 'Blue', 'Green', 'Red', 'White')
RARITIES = ('Common', 'Uncommon', 'Rare', 'Mythic Rare', 'Special')
CARD_TYPES = ('Creature', 'Artifact', 'Enchantment', 'Sorcery', 'Instant', 'Land', 'Planeswalker')
PROPERTIES = ('flying', 'trample', 'unblockable', 'indestructible', 'defender', 'flash', 'first strike', 'fear',
'lifelink', 'deathtouch', 'infect', 'wither', 'haste', 'flanking', 'shroud', 'hexproof', 'vigilance', 'persist', 'undying', 'exalted', 'reach', 'intimidate')

Card = namedtuple('Card', (
    'date',
    'cmc',
    'is_common',
    'is_uncommon',
    'is_rare',
    'is_mythic_rare',
    'is_special',
    'is_black',
    'is_blue',
    'is_green',
    'is_red',
    'is_white',
    'is_creature',
    'is_artifact',
    'is_enchantment',
    'is_sorcery',
    'is_instant',
    'is_land',
    'is_planeswalker',
    'power',
    'toughness',
    'loyalty',
    #'has_flying',
    #'has_trample',
    #'has_indestructible',
    #'has_unblockable',
    #'has_defender'
    ) + tuple('has_' + x.lower().replace(' ', '_') for x in PROPERTIES)

    )



def read_allsets(filename):

    # Load JSON to a dict/list structure
    with open(filename, 'rb') as f:
        sets_ = json.load(f)

    # Extract relevant features from the card dicts and store them
    # as list of namedtuples
    l = list()
    for k, set_ in sets_.items():
        date = datetime.strptime(set_['releaseDate'], '%Y-%m-%d')
        print('{} {}'.format(k, set_['name']))

        for card in set_['cards']:
            if card['layout'] != 'normal':
                continue
            if 'Creature' not in card['types']:
                continue

            d = {f: False for f in Card._fields}

            # Analyze card text if it consists only of simple
            # attributes, otherwise, ignore this card

            # TODO: cut out text in parenthesis
            text = card.get('text', '')
            text = re.sub(r'\([^(]*\)', '', text)
            properties = set([x.lower().strip() for x in re.split(',\s*|\n', text)])
            if not set(properties).issubset(set(PROPERTIES)):
                leftover = set(properties) - set(PROPERTIES)
                for x in leftover:
                    if ' ' not in x:
                        print(x)
                continue
            for p in properties:
                d['has_' + p.lower().replace(' ', '_')] = True

            # Get other attributes from card dict

            d['date'] = date
            d['cmc'] = int(card.get('cmc', 0))
            d['power'] = int(card.get('power', 0))
            d['toughness'] = int(card.get('toughness', 0))


            d['loyalty'] = int(card.get('loyalty', 0))

            assert card['rarity'] in RARITIES, card['rarity']
            d['is_' + card['rarity'].lower().replace(' ', '_')] = True

            colors = set(card.get('colors', ()))
            assert colors.issubset(set(COLORS)), colors
            for color in colors:
                d['is_' + color.lower()] = True

            assert set(card['types']).issubset(set(CARD_TYPES)), card['types']
            for type_ in card['types']:
                d['is_' + type_.lower()] = True

            l.append(Card(**d))

    df = pd.DataFrame(l, columns = Card._fields)
    print(df)


if __name__ == '__main__':
    import sys

    read_allsets(sys.argv[1])

