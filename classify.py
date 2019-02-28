import os
from main import Analyser


class Classify(object):
    CURRENT_DIR = os.path.dirname(__file__) + '/'

    def __init__(self):
        if Classify.CURRENT_DIR is '/':
            Classify.CURRENT_DIR = ''

        return None

    def start(self):
        instance = Analyser()

        profiles = instance.load_data('profiles.json')
        classified_profiles = instance.load_data('classified_profiles.json')

        try:
            for key in profiles.keys():
                profile = profiles[key]

                if key in classified_profiles:
                    continue

                print "\nUsername: {}\nFull name: {}\nBio: {}\nFollowers: {}, Following: {}, Posts: {}\nGhost: {}, Relationship: {}".format(profile['username'],
                    profile['full_name'].encode('ascii', 'replace'),
                    profile['biography'].encode('ascii', 'replace'),
                    profile['follower_count'],
                    profile['following_count'],
                    profile['media_count'],
                    profile['is_ghost'],
                    profile['relationship'])

                valid_input = False
                while not valid_input:
                    is_authentic = raw_input('Is authentic? (a/d): ')

                    if is_authentic == 'a':
                        print 'Selected valid.'
                        profile['is_authentic'] = True
                        valid_input = True

                        classified_profiles[key] = profile
                    elif is_authentic == 'd':
                        print 'Selected invalid.'
                        profile['is_authentic'] = False
                        valid_input = True

                        classified_profiles[key] = profile
                    else:
                        print 'Invalid input entered -- try again.'

        except KeyboardInterrupt as kbi:
            print 'KeyboardInterrupt: {}'.format(str(kbi))
        except Exception as e:
            print 'Exception: {}'.format(str(e))
                    
        instance.save_data('classified_profiles.json', classified_profiles)

if __name__ == '__main__':

    instance = Classify()

    instance.start()