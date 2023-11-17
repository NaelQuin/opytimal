'''
Module of the tests proccessment functions
'''

__all__ = ['testLoop']

def testLoop(globals, locals=None, title=None):
    if title is not None:
        print(
            '======================================',
            f'{title.title()} interaction zone begin',
            '======================================',
            sep='\n'
            )
    while True:
        try:
            code = input('\nType your test code: ')

            if code.lower() in ['q', 'exit']:
                raise KeyboardInterrupt

            if any(code):
                if '=' in code and '==' not in code:
                    exec(code, globals, locals)
                elif code.isalpha():
                    value = eval(code, globals, locals)
                    print('\n\t', f'{code} = {value}',
                          end='\n')
                else:
                    print('\n\t', eval(code, globals, locals), end='\n')
            else:
                print('\n\n*** Nothing typed *** Press CTRL+C to skip ***\n')

        except (KeyboardInterrupt, EOFError):
            # Break a line
            print()

            # Stop the loop
            break

        except Exception as err:
            # Show the error
            print(f'\n{err.__class__.__name__}: {err}\n')

    if title is not None:
        print(
            '====================================',
            f'{title.title()} interaction zone end',
            '====================================',
            sep='\n'
            )

    return None