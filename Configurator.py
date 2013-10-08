import inspect

def make_configurator (func, configurator):

    args = inspect.getfullargspec(func).args

    try:
        while True:
            name = next(configurator)
            config = { arg: configurator.gi_frame.f_locals[arg] for arg in args }
            print ('Running {}'.format(name))
            func (**config)
    except StopIteration:
        pass

def configurator_for (func):
    def run_configuration (configurator):
        make_configurator (func, configurator())
    return run_configuration

# Test code

def testfunc (**kwargs):
    print (locals())

@configurator_for(testfunc)
def cfg ():
    for a in ['Hello', 'World']:
        for b in ['Foo', 'Bar']:
            yield '{} {}'.format(a, b)
