import os
from collections import OrderedDict
import pyparsing as p


class Project(object):
    """
    This class represents a GRASP project.

    It currently can only create a blank project.
    """

    extensions = {9: 'g9p', 10: 'gxp'}

    def __init__(self, name, version='10.0.1'):
        self.name = str(name)
        self.version = str(version)
        self.tor = ObjectRepository()
        self.tci = CommandInterface()

    def create(self, folder):
        """
        Create a new project folder and a working folder, and save the
        project, object, and command files.
        """
        base = os.path.abspath(os.path.join(folder, self.name))
        working = os.path.join(base, 'working')
        os.mkdir(base)
        os.mkdir(working)
        major_version = int(self.version.split('.')[0])
        project_file = os.path.join(base, '{}.{}'.format(self.name,
                                                         self.extensions[major_version]))
        with open(project_file, 'w') as f:
            f.write(str(self))
        tor_file = os.path.join(working, '{}.tor'.format(self.name))
        with open(tor_file, 'w') as f:
            f.write(str(self.tor))
        tci_file = os.path.join(working, '{}.tci'.format(self.name))
        with open(tci_file, 'w') as f:
            f.write(str(self.tci))

    
    def __str__(self):
        lines = ['[Comment]',
                 'Project comment',
                 '[TOR file]',
                 os.path.join('working', '{}.tor'.format(self.name)),
                 '[Auxiliary TOR files]',
                 '[TCI file]',
                 os.path.join('working', '{}.tci'.format(self.name)),
                 '[Default units]',
                 'GHz m S/m 1',
                 '[Project setup]',
                 '<!DOCTYPE Project>',
                 '<Project version="{}" application="GRASP">'.format(self.version),
                 ' <Results/>',
                 ' <ResultWindows/>',
                 ' <WizardData/>',
                 ' <view_configuration/>',
                 '</Project>']
        return ''.join(['{}\n'.format(line) for line in lines])


class Grammar(object):
    """
    This class contains the basic grammar for the TICRA
    .tor and .tci file formats.
    """

    plus_or_minus = p.Literal('+') ^ p.Literal('-')
    number = p.Combine(p.Optional(plus_or_minus) +
                       p.Word(p.nums) +
                       p.Optional('.' + p.Word(p.nums)) +
                       p.Optional(p.CaselessLiteral('E') +
                                  p.Word(p.nums + '+-', p.nums)))
    quantity = number + p.Optional(p.Word(p.alphas, p.alphanums + '-^/' + ' '), default=None)
    quantity.setParseAction(lambda tokens: Quantity(tokens[0], tokens[1]))
    identifier = p.Word(p.alphas + '_', p.alphanums + '_')
    value = p.Forward()
    elements = p.delimitedList(value)
    member = p.Group(identifier + p.Suppress(':') + value)
    members = p.delimitedList(member)
    ref = 'ref' + p.Suppress('(') + identifier('ref') + p.Suppress(')')
    ref.setParseAction(lambda tokens: [Ref(tokens['ref'])])
    sequence = 'sequence' + \
               p.Suppress('(') + \
               p.Optional(elements)('elements') + \
               p.Suppress(')')
    sequence.setParseAction(lambda tokens: [Sequence(tokens.get('elements', []))])
    struct = 'struct' + \
             p.Suppress('(') + \
             p.Optional(members)('members') + \
             p.Suppress(')')
    struct.setParseAction(lambda tokens: [Struct(list(tokens.get('members', [])))])
    comment = p.QuotedString('"', unquoteResults=False)
    # This could be a filename or just a string.
    other = p.Word(p.alphanums + '/._-')
    value << (quantity | ref | sequence | struct | comment | other)


class CommandInterface(OrderedDict, Grammar):
    """
    This class represents a TICRA Command Interface (.tci) file.
    """

    # Add logic for reading and writing batch commands.

    def __init__(self, other={}):
        super(CommandInterface, self).__init__(other)
        self.files_read_all = []
    
    def parse(self, string):
        # Improve this if possible.
        files_read_all = p.Suppress('files read all ') + p.restOfLine
        files_read_all.setParseAction(lambda tokens: self.files_read_all.append(tokens.pop()))
        format = p.ZeroOrMore(files_read_all) + \
                 p.ZeroOrMore(Command.format) + \
                 p.Suppress('QUIT') + p.StringEnd()
        format.ignore(p.pythonStyleComment)
        return format.parseString(string)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.update([(number+1, command) for number, command in enumerate(self.parse(f.read()))])

    def save(self, filename, batch_mode=False):
        """
        Write all the class descriptions to a file, which should have
        the .tci extension.
        """
        string = str(self)
        if batch_mode:
            batch_commands = '\n'.join(['files_read_all {}'.format(tor) for tor in self.files_read_all])
            string = '\n'.join([batch_commands, string])
        with open(filename, 'w') as f:
            f.write(string)

    def __str__(self):
        return '\n\n'.join(['{} cmd_{}'.format(command, index) for index, command in self.iteritems()] + ['QUIT'])

    # This method is identical between CommandInterface and ObjectRepository.
    # This could be a class method or a superclass method.
    def traverse(self, name, thing, action, filter):
        if filter(name, thing):
            action(name, thing)
        elif isinstance(thing, Sequence):
            for index, element in enumerate(thing):
                self.traverse(index, element, action, filter)
        elif isinstance(thing, OrderedDict):
            for key, value in thing.iteritems():
                self.traverse(key, value, action, filter)

    def reference(self, repository, names_and_things=None):
        """
        Update references to classes by replacing object names with
        the actual objects from ObjectRepository repository.
        """
        if names_and_things is None:
            names_and_things = self.iteritems()
        for name, thing in names_and_things:
            self.traverse(name,
                          thing,
                          lambda name, thing: thing.set(repository[thing.name]),
                          lambda name, thing: isinstance(thing, Ref))
                
    def dereference(self, names_and_things=None):
        """
        Update references to classes by replacing objects with their names.
        """
        if names_and_things is None:
            names_and_things = self.iteritems()
        for name, thing in names_and_things:
            self.traverse(name,
                          thing,
                          lambda name, thing: thing.set(thing.name),
                          lambda name, thing: isinstance(thing, Ref))


class ObjectRepository(OrderedDict, Grammar):
    """
    This class represents a TICRA Object Repository (.tor) file.
    """

    def parse(self, string):
        format = p.ZeroOrMore(Physical.format) + p.StringEnd()
        format.ignore(p.cppStyleComment)
        #format.ignore(p.pythonStyleComment)
        return format.parseString(string)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.update([(physical.display_name, physical) for physical in self.parse(f.read())])
            
    def save(self, filename):
        """
        Write all the class descriptions to a file, which should have
        the .tor extension.
        """
        with open(filename, 'w') as f:
            f.write(str(self))

    def __str__(self):
        return '\n \n'.join([str(physical) for physical in self.values()]) + '\n'

    # This method is identical between CommandInterface and ObjectRepository.
    def traverse(self, name, thing, action, filter):
        if filter(name, thing):
            action(name, thing)
        elif isinstance(thing, Sequence):
            for index, element in enumerate(thing):
                self.traverse(index, element, action, filter)
        elif isinstance(thing, OrderedDict):
            for key, value in thing.iteritems():
                self.traverse(key, value, action, filter)

    def reference(self, names_and_things=None):
        """
        Update references to classes by replacing object names with the actual objects.
        """
        if names_and_things is None:
            names_and_things = self.iteritems()
        for name, thing in names_and_things:
            self.traverse(name,
                          thing,
                          lambda name, thing: thing.set(self[thing.name]),
                          lambda name, thing: isinstance(thing, Ref))
                
    def dereference(self, names_and_things=None):
        """
        Update references to classes by replacing objects with their names.
        """
        if names_and_things is None:
            names_and_things = self.iteritems()
        for name, thing in names_and_things:
            self.traverse(name,
                          thing,
                          lambda name, thing: thing.set(thing.name),
                          lambda name, thing: isinstance(thing, Ref))

    
class Command(OrderedDict):
    """
    This class is a lightweight container for GRASP commands.
    
    It inherits from OrderedDict so that it remembers the ordering of its properties.
    """
    
    format = p.Suppress('COMMAND') + \
             p.Suppress('OBJECT') + \
             Grammar.identifier('target_name') + \
             Grammar.identifier('command_name') + \
             p.Suppress('(') + \
             p.Optional(Grammar.members)('members') + \
             p.Suppress(')') + \
             p.CaselessLiteral('cmd_').suppress() + \
             p.Word(p.nums)('number')
    format.ignore(p.cppStyleComment) # '// comment'
    format.ignore(p.pythonStyleComment) # '# comment'
    format.ignore(p.Literal('&'))
    format.setParseAction(lambda tokens: [Command(list(tokens.get('members', [])), tokens['target_name'], tokens['command_name'])])

    def __init__(self, other={}, target_name=None, command_name=None):
        super(Command, self).__init__(other)
        self.target_name = str(target_name)
        self.command_name = str(command_name)

    def __str__(self):
        lines = ['COMMAND OBJECT {} {}'.format(self.target_name, self.command_name)]
        lines.append('(')
        for k, v in self.iteritems():
            lines.append('  {:16s} : {},'.format(k, v))
        lines[-1] = lines[-1].rstrip(',')
        lines.append(')')
        return ' &\n'.join(lines)
        
    def __repr__(self):
        return '{}({{{}}}, {!r}, {!r})'.format(self.__class__.__name__, ', '.join(['{!r}: {!r}'.format(k, v) for k, v in self.iteritems()]), self.target_name, self.command_name)


class Physical(OrderedDict):
    """
    This class is a lightweight container for GRASP physical objects.

    It inherits from OrderedDict so that it remembers the ordering of its properties.
    """
    format = Grammar.identifier('display_name') + \
             Grammar.identifier('class_name') + \
             p.Suppress('(') + \
             p.Optional(Grammar.members)('members') + \
             p.Suppress(')')
    format.ignore(p.cppStyleComment) # '// comment'
    format.ignore(p.pythonStyleComment) # '# comment'
    format.setParseAction(lambda tokens: [Physical(list(tokens.get('members', [])), tokens['display_name'], tokens['class_name'])])

    def __init__(self, other={}, display_name=None, class_name=None):
        super(Physical, self).__init__(other)
        self.display_name = str(display_name)
        self.class_name = str(class_name)

    def __str__(self):
        lines = ['{}  {}  '.format(self.display_name, self.class_name)]
        lines.append('(')
        for k, v in self.iteritems():
            lines.append('  {:16s} : {},'.format(k, v))
        lines[-1] = lines[-1].rstrip(',')
        lines.append(')')
        return '\n'.join(lines)
    
    def __repr__(self):
        return '{}({{{}}}, {!r}, {!r})'.format(self.__class__.__name__, ', '.join(['{!r}: {!r}'.format(k, v) for k, v in self.iteritems()]), self.display_name, self.class_name)


class Ref(object):
    """
    This class is a wrapper for Physical objects. It stops the
    recursion of __str__ and __repr__ to avoid infinite loops.
    """

    def __init__(self, ref):
        self.set(ref)

    def set(self, ref):
        """
        This allows self.ref to be changed in a lambda statement.
        """
        if isinstance(ref, str):
            self.name = ref
        elif isinstance(ref, Physical):
            self.name = ref.display_name
        else:
            raise ValueError("Invalid Ref: {!r}".format(ref))
        self.ref = ref
        
    def __str__(self):
        return 'ref({})'.format(self.name)

    def __repr__(self):
        return self.__str__()


class Struct(OrderedDict):
    
    def __str__(self):
        return 'struct({})'.format(', '.join('{}: {}'.format(k, v) for k, v in self.iteritems()))

    def __repr__(self):
        return '{}({{{}}})'.format(self.__class__.__name__, ', '.join(['{!r}: {!r}'.format(k, v) for k, v in self.iteritems()]))


class Sequence(list):
    
    def __str__(self):
        return 'sequence({})'.format(', '.join(str(s) for s in self))

    def __repr__(self):
        return '{}([{}])'.format(self.__class__.__name__, ', '.join([repr(s) for s in self]))


class Quantity(object):
    """
    This class is a lightweight wrapper for a physical quantity that
    may carry units.
    """

    def __init__(self, number, units=None):
        if isinstance(number, str):
            try:
                self.number = int(number)
            except ValueError:
                self.number = float(number)
        # This should possibly be expanded for numpy number types.
        elif isinstance(number, int) or isinstance(number, float):
            self.number = number
        else:
            raise ValueError("Invalid number: {!r}".format(number))
        # Add logic for allowed units.
        self.units = units

    def __str__(self):
        # This uses a lowercase e for exponential notation; GRASP writes
        # an uppercase E, but can read both. Using repr instead of str
        # seems to ensure that no rounding occurs.
        if self.units is None:
            return '{!r}'.format(self.number)
        else:
            return '{!r} {}'.format(self.number, self.units)

    def __repr__(self):
        if self.units is None:
            return '{}({})'.format(self.__class__.__name__, self.number)
        else:
            return '{}({!r}, {!r})'.format(self.__class__.__name__, self.number, self.units)
