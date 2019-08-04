#!/usr/bin/env python3

from collections import OrderedDict, namedtuple, Counter
import json
import logging
from datetime import datetime
import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(sh)

COLOR_DECODE = {
    'U': 'blue',
    'B': 'black',
    'G': 'green',
    'R': 'red',
    'W': 'white',
    'S': 'snow',
    }

COLORS = tuple(COLOR_DECODE.values())


RARITIES = ('common', 'uncommon', 'rare', 'mythic', 'special')
CARD_TYPES = ('Creature', 'Artifact', 'Enchantment', 'Sorcery', 'Instant', 'Land', 'Planeswalker')
PROPERTIES = (
    'deathtouch',
    'defender',
    'exalted',
    'fear',
    'first strike',
    'flanking',
    'flash',
    'flying',
    'haste',
    'hexproof',
    'indestructible',
    'infect',
    'intimidate',
    'lifelink',
    'persist',
    'reach',
    'shadow',
    'shroud',
    'trample',
    'unblockable',
    'undying',
    'vigilance',
    'wither',
)

identifier = lambda s: s.lower().replace(' ', '_')

Card = namedtuple('Card', (
    'date',
    'cmc',
    'is_common',
    'is_uncommon',
    'is_rare',
    'is_mythic',
    'is_special',
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
    ) \
    + tuple('has_' + identifier(x) for x in PROPERTIES)
    + tuple('abs_devotion_' + identifier(x) for x in COLORS)
    + tuple('rel_devotion_' + identifier(x) for x in COLORS)

    )



def read_allsets(filename):

    # Load JSON to a dict/list structure
    with open(filename, 'r') as f:
        sets_ = json.load(f, object_pairs_hook = OrderedDict)

    # Extract relevant features from the card dicts and store them
    # as list of namedtuples
    l = list()
    names = list()
    for k, set_ in sets_.items():
        # The fun editions make all kinds of messy stuff
        if k in ('UNH', 'UST'):
            continue

        date = datetime.strptime(set_['releaseDate'], '%Y-%m-%d')
        print('{} {}'.format(k, set_['name']))

        for card in set_['cards']:

            if card['name'] in names:
                logger.debug('ignoring "{}" because we seen a card with the same name before'.format(card['name']))
                continue

            # Exclude non-normal and non-create cards (for now)

            if card['layout'] != 'normal':
                logger.debug('ignoring "{}" because its layout "{}" is ignored.'.format(card['name'], card['layout']))
                continue
            if 'Creature' not in card['types']:
                logger.debug('ignoring "{}" because its not a creature.'.format(card['name']))
                continue

            d = {f: False for f in Card._fields}

            # Analyze card text if it consists only of simple
            # attributes, otherwise, ignore this card

            text = card.get('text', '')
            text = re.sub(r'\([^(]*\)', '', text)
            properties = set([x.lower().strip() for x in re.split(',\s*|\n', text)]) - set([""])
            if not set(properties).issubset(set(PROPERTIES)):
                leftover = set(properties) - set(PROPERTIES)
                for x in leftover:
                    if ' ' not in x:
                        logger.debug('ignoring "{}" because property "{}" not understood.'.format(card['name'], x))
                continue

            # print(card['name'])

            for p in properties:
                d['has_' + p.lower().replace(' ', '_')] = True

            # Get other attributes from card dict

            d['date'] = date
            d['cmc'] = int(card.get('cmc', 0))

            pwr = card.get('power', 0)
            d['power'] = 99999 if pwr == '∞' else int(pwr)
            d['toughness'] = int(card.get('toughness', 0))
            d['loyalty'] = int(card.get('loyalty', 0))

            # Rarity
            card['rarity'] = card['rarity'].lower()

            assert card['rarity'] in RARITIES, card['rarity']
            d['is_' + identifier(card['rarity'])] = True

            # Parse mana cost
            # ...into devotion

            if 'manaCost' not in card:
                # There are actually cards that are creatures, have a color
                # but no mana cost (eg. because they are also lands such as Dryad Arbor),
                # ignore them
                logger.debug('Ignoring card "{}" because it has no manaCost'.format(card['name']))
                continue

            def devotion_colors():
                # Find all the {...} thingies
                matches = re.findall(r'{([^}]+)}', card['manaCost'])
                for m in matches:
                    # Do we have something like {R/G}?
                    split = re.match(r'([BWRGU])/([BWRGUP])', m)
                    if split is not None:
                        yield COLOR_DECODE[split.groups()[0]]
                        if split.groups()[1] != 'P':
                            yield COLOR_DECODE[split.groups()[1]]

                    elif m.isdigit():
                        # Amount of colorless (don't care) mana to be paid,
                        # unimportant for devotion, just ignore it.
                        pass

                    else:
                        yield COLOR_DECODE[m]


            c = { color: 0 for color in COLORS }
            for color in devotion_colors():
                c[color] += 1

            if c['snow'] != 0: # Ignore snow mana cards for now
                continue

            for k, v in c.items():
                d['abs_devotion_' + k] = v 
                d['rel_devotion_' + k] = v / d['cmc'] if d['cmc'] else 0

            # Types

            assert set(card['types']).issubset(set(CARD_TYPES)), card['types']
            for type_ in card['types']:
                d['is_' + type_.lower()] = True

            names.append(card['name'])
            l.append(Card(**d))

    logger.info('Read {} cards.'.format(len(l)))

    df = pd.DataFrame(l, columns = Card._fields)
    return df, names


if __name__ == '__main__':
    import sys

    read_allsets(sys.argv[1])

