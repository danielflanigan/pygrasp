import os
from copy import deepcopy
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
    This class contains the pyparsing grammar for the TICRA .tor and
    .tci file formats.
    """

    # It might be cleaner to have the grammar for each class in the
    # class itself, but this causes scope problems.
    plus_or_minus = p.Literal('+') ^ p.Literal('-')
    number = p.Combine(p.Optional(plus_or_minus) +
                       p.Word(p.nums) +
                       p.Optional('.' + p.Word(p.nums)) +
                       p.Optional(p.CaselessLiteral('E') +
                                  p.Word(p.nums + '+-', p.nums)))
    quantity = number + p.Optional(p.Word(p.alphas, p.alphanums + '-^/' + ' '), default=None)
    quantity.setParseAction(lambda tokens: Quantity(tokens[0], tokens[1]))
    # Added '.' to handle Brad's EBEX sims. See if this breaks anything.
    # An identifier is no longer a valid Python variable.
    identifier = p.Word(p.alphas + '_', p.alphanums + '_.')
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
    # This could be a filename or just a string. Added '\' to handle EBEX sim.
    # Should convert to unix filename.
    other = p.Word(p.alphanums + '\/._-')
    value << (quantity | ref | sequence | struct | comment | other)

    physical = (identifier('display_name') +
                identifier('class_name') +
                p.Suppress('(') +
                p.Optional(members)('members') +
                p.Suppress(')'))
    physical.ignore(p.cppStyleComment) # '// comment'
    physical.ignore(p.pythonStyleComment) # '# comment'
    physical.setParseAction(lambda tokens: [Physical(tokens['display_name'],
                                                     tokens['class_name'],
                                                     list(tokens.get('members', [])))])

    object_repository = p.ZeroOrMore(physical) + p.StringEnd()
    object_repository.ignore(p.cppStyleComment)
    object_repository.ignore(p.pythonStyleComment)

    command = (p.Suppress('COMMAND') +
               p.Suppress('OBJECT') +
               identifier('target_name') +
               identifier('command_name') +
               p.Suppress('(') +
               p.Optional(members)('members') +
               p.Suppress(')') +
               p.CaselessLiteral('cmd_').suppress() +
               p.Word(p.nums)('number'))
    command.ignore(p.cppStyleComment) # '// comment'
    command.ignore(p.pythonStyleComment) # '# comment'
    command.ignore(p.Literal('&'))
    command.setParseAction(lambda tokens: [Command(tokens['target_name'],
                                                   tokens['command_name'],
                                                   list(tokens.get('members', [])))])

    # Add support for other batch commands.
    batch_command = p.CaselessLiteral('FILES READ ALL') + other + p.LineEnd().suppress()
    batch_command.setParseAction(lambda tokens: [BatchCommand(tokens)])
    quit_command = p.CaselessLiteral('QUIT')

    # Add support for multiple QUIT statements.
    command_interface = (p.ZeroOrMore(batch_command)('batch_commands') +
                         p.ZeroOrMore(command)('commands') +
                         quit_command +
                         p.StringEnd())
    command_interface.ignore(p.cppStyleComment)
    command_interface.ignore(p.pythonStyleComment)


class CommandInterface(list):
    """
    This class represents a TICRA Command Interface (.tci) file.
    """

    def __init__(self, other=[]):
        super(CommandInterface, self).__init__(other)
        self.batch_commands = []    
    
    def load(self, filename):
        with open(filename, 'r') as f:
            parsed = Grammar.command_interface.parseFile(f)
            self.batch_commands.extend(parsed.get('batch_commands', []))
            self.extend(parsed.get('commands', []))

    def save(self, filename, batch_mode=True):
        """
        Write all the class descriptions to a file, which should have
        the .tci extension.
        """
        with open(filename, 'w') as f:
            if batch_mode:
                f.write('\n'.join([str(bc) for bc in self.batch_commands]) + '\n\n' + str(self))
            else:
                f.write(str(self))

    def __str__(self):
        return '\n\n'.join(['{} cmd_{}'.format(command, index + 1) for index, command in enumerate(self)] + ['QUIT'])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               super(CommandInterface, self).__repr__())


class ObjectRepository(OrderedDict):
    """
    This class represents a TICRA Object Repository (.tor) file.
    """

    def load(self, filename):
        with open(filename, 'r') as f:
            self.update([(physical.display_name, physical)
                         for physical in Grammar.object_repository.parseFile(f)])

    def save(self, filename):
        """
        Write all the class descriptions to a file, which should have
        the .tor extension.
        """
        with open(filename, 'w') as f:
            f.write(str(self))

    def __str__(self):
        return '\n \n'.join([str(physical) for physical in self.values()]) + '\n'

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               super(OrderedDict, self).__repr__())

    def extract(self, name, source, careful=False):
        """
        Add to this instance the named object from the given source
        and all references on which it depends, all the way up the
        tree, overwriting previous objects.

        If careful is False, the traversal will follow all references
        and add the objects to this instance until it encounters only
        objects that have no dependencies, replacing objects that have
        the same name as objects already in this instance. If an
        object dependency is cyclic, the traversal will lead to an
        infinite loop. This should never happen, and it's not clear
        whether GRASP even allows this. (This could happen most easily
        with coordinate systems.)

        If careful is True, the traversal will stop when it encounters
        an object whose name is already a key in this instance, and it
        will not overwrite this object. Using this option with a name
        already in the dictionary has no effect. This avoids an
        infinite loop in the case of a cyclic reference. It is also
        more efficient, since it avoids repeatedly traversing the
        tree. If there are objects in the tree whose references have
        not been imported, a careful traversal will not import these
        references unless other imported objects reference them.

        A repository populated only by calls to this method with
        careful=True should contain all necessary references.
        """
        obj = source[name]
        if name in self and careful:
            pass
        else:
            self[obj.display_name] = obj
            obj.traverse(lambda name, thing: isinstance(thing, Ref),
                         lambda name, thing: self.extract(thing.name, source, careful))


class Command(OrderedDict):
    """
    This class is a container for GRASP commands.
    
    It inherits from OrderedDict so that it remembers the ordering of its properties.
    """
    
    def __init__(self, target_name, command_name, other={}):
        super(Command, self).__init__(other)
        self.target_name = str(target_name)
        self.command_name = str(command_name)

    def __str__(self):
        lines = ['COMMAND OBJECT {} {}'.format(self.target_name, self.command_name)]
        lines.append('(')
        for k, v in self.items():
            lines.append('  {:16s} : {},'.format(k, v))
        lines[-1] = lines[-1].rstrip(',')
        lines.append(')')
        return ' &\n'.join(lines)
        
    def __repr__(self):
        return '{}({!r}, {!r}, {{{}}})'.format(self.__class__.__name__,
                                               self.target_name,
                                               self.command_name,
                                               ', '.join(['{!r}: {!r}'.format(k, v) for k, v in self.items()]))

    # This code is shared between Command and Physical objects. Fix this.
    def traverse(self, test, action):
        """
        Recursively visit all members of this object. See visit() for
        parameter meanings.
        """
        for name, thing in self.items():
            self.visit(name, thing, test, action)
        
    def visit(self, name, thing, test, action):
        """
        Recursively visit every member of this object, calling
        action(name, thing) if test(name, thing) is True.
        """
        if test(name, thing):
            action(name, thing)
        elif isinstance(thing, Sequence):
            for index, element in enumerate(thing):
                self.visit(index, element, test, action)
        elif isinstance(thing, Struct):
            for key, value in thing.items():
                self.visit(key, value, test, action)


class BatchCommand(list):

    def __str__(self):
        return ' '.join(self)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               super(BatchCommand, self).__repr__())


class Physical(OrderedDict):
    """
    This class is a lightweight container for GRASP physical objects.

    It inherits from OrderedDict so that it remembers the ordering of its properties.
    """

    def __init__(self, display_name, class_name, other={}):
        super(Physical, self).__init__(other)
        self.display_name = str(display_name)
        self.class_name = str(class_name)

    def __str__(self):
        lines = ['{}  {}  '.format(self.display_name, self.class_name)]
        lines.append('(')
        for k, v in self.items():
            lines.append('  {:16s} : {},'.format(k, v))
        lines[-1] = lines[-1].rstrip(',')
        lines.append(')')
        return '\n'.join(lines)
    
    def __repr__(self):
        return '{}({!r}, {!r}, {{{}}})'.format(self.__class__.__name__, 
                                               self.display_name,
                                               self.class_name,
                                               ', '.join(['{!r}: {!r}'.format(k, v) for k, v in self.items()]))

    # This code is shared between Command and Physical objects. Fix this.
    def traverse(self, test, action):
        """
        Recursively visit all members of this object. See visit() for
        parameter meanings.
        """
        for name, thing in self.items():
            self.visit(name, thing, test, action)
        
    def visit(self, name, thing, test, action):
        """
        Recursively visit every member of this object, calling
        action(name, thing) if test(name, thing) is True.
        """
        if test(name, thing):
            action(name, thing)
        elif isinstance(thing, Sequence):
            for index, element in enumerate(thing):
                self.visit(index, element, test, action)
        elif isinstance(thing, Struct):
            for key, value in thing.items():
                self.visit(key, value, test, action)


class Ref(object):
    """
    This class is a wrapper for Physical objects. It stops the
    recursion of __str__ to avoid infinite loops.

    To do: if pointers to actual objects become useful, implement
    __repr__() to stop infinite loops.
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

    # This will not work for referenced objects, only strings.
    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__,
                                 self.ref)


class Struct(OrderedDict):
    
    def __str__(self):
        return 'struct({})'.format(', '.join('{}: {}'.format(k, v) for k, v in self.items()))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                                   super(OrderedDict, self).__repr__())


class Sequence(list):

    def __str__(self):
        return 'sequence({})'.format(', '.join(str(s) for s in self))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               super(Sequence, self).__repr__())


class Quantity(object):
    """
    This class represents a physical quantity that may or may not
    carry units.
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
        # This uses a lowercase e for exponential notation; GRASP
        # writes an uppercase E, but can read either. Using repr
        # instead of str for numbers seems to ensure that no rounding
        # occurs.
        if self.units is None:
            return '{!r}'.format(self.number)
        else:
            return '{!r} {}'.format(self.number, self.units)

    def __repr__(self):
        if self.units is None:
            return '{}({!r})'.format(self.__class__.__name__, self.number)
        else:
            return '{}({!r}, {!r})'.format(self.__class__.__name__, self.number, self.units)


