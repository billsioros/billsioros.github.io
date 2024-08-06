---
title: "Let's Talk Python Metaclasses"
date: 2024-04-17
draft: false
colab: "https://colab.research.google.com/github/billsioros/billsioros.github.io/blob/master/static/code/python_metaclasses.ipynb"
author: "Vassilis Sioros"
categories:
  - Programming Patterns
tags:
  - Python
  - Programming Patterns
image: /images/posts/metaclasses.png
thumbnail: /images/posts/metaclasses.thumbnail.png
description: "Where Classes Get Their Magic!"
toc:
---


Python metaclasses give you control over class creation and behavior. They act as blueprints, shaping inheritance and behavior. Just as classes govern instances, metaclasses oversee classes.



## A little know gem

Ever heard of a metaclass called `abc.ABC`? Chances are you've been using it without even realizing it. This little gem is pretty handy for crafting abstract base classes (ABCs) in Python. It's like the rulebook that says, *"Hey, if you're a subclass, you gotta follow these rules and implement these methods."*

Take a look at `Vehicle` — it's an abstract base class setting the stage. And here comes `Car`, ready to roll as a subclass. But hey, to be a proper `Car`, it's gotta have that `move` method implemented. Thanks to `abc.ABC`, Python keeps everyone in line, making sure subclasses play by the rules.


```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def move(self):
        pass

class Car(Vehicle):
    def move(self):
        return "Car is moving"
```

## Let's Dive Deeper!

Metaclasses often roll out the red carpet for the `__new__` and `__init__` magic methods, letting you customize class creation like a pro.

- `__new__` is like the early bird of class creation, chirping before the class is even born. It's the one responsible for bringing the class object into existence. Here's where you can tweak base classes, but watch out for Method Resolution Order (MRO) issues if you're not careful.
- Now, `__init__` strolls in fashionably late, after the class has been born. Its job is to spruce up the class object once it's out there in the world. When `__init__` arrives, the class's namespace is already filled with goodies, like attributes and all. So, it's your chance to add those final touches, knowing that the class is ready to rock and roll.


```python
class Meta(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        print("Meta.__new__ called with metaclass: %s" % cls.__name__)

        # Call the __new__ method of the superclass (type) to create the class
        return super().__new__(cls, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        print("Meta.__init__ called for class: %s" % cls.__name__)

        # Display class-level attributes
        print(
            "  Class attributes for %s: %s"
            % (cls.__name__, getattr(cls, "class_attribute", None))
        )

        # Call the __init__ method of the superclass (type) to initialize the class
        super().__init__(name, bases, attrs)

class Foo(metaclass=Meta):
    pass
```

    Meta.__new__ called with metaclass: Meta
    Meta.__init__ called for class: Foo
      Class attributes for Foo: None


## Our first attempt


```python
class UppercaseAttributeMeta(type):
    def __new__(cls, name, bases, dct):
        uppercase_attrs = {
            attr.upper(): value
            for attr, value in dct.items()
            if not attr.startswith("__")  # Exclude magic methods
        }
        return super().__new__(cls, name, bases, uppercase_attrs)

class Bar(metaclass=UppercaseAttributeMeta):
    message = "hello"

print(Bar.MESSAGE)
```

    hello


So, what's happening here?

We're creating a metaclass called `UppercaseAttributeMeta`. When we create a class and set its metaclass to `UppercaseAttributeMeta`, it's like telling Python, "Hey, when you create this class, make sure all the attribute names are in uppercase!"

Inside our metaclass, we have a special method called `__new__`. This method is like the constructor for our class. It gets called when Python is creating a new class. Inside `__new__`, we're basically saying:

1. "Hey, Python, give me all the attributes (`attr`) and their values (`value`) that are inside this class (`dct`)."
2. "Now, let's check each attribute. If it's not a special magic method (like `__init__`), let's convert its name to uppercase and store it in a new dictionary (`uppercase_attrs`)."
3. "Okay, Python, go ahead and create the class for me, but use this new dictionary with the uppercase attribute names and their values."

Then, we define a class called `Bar` and tell Python to use our `UppercaseAttributeMeta` metaclass. So, any attributes we define in `Bar` will automatically have their names converted to uppercase.

## JSON Made Easy

Imagine you're developing an application that interacts heavily with JSON data. As part of your project, you find yourself repeatedly defining classes to represent different JSON structures. This manual process not only consumes time but also increases the likelihood of errors creeping into your codebase.

Your objective is to devise a solution that automates the creation of these classes based on JSON schemas. By doing so, you aim to ensure that the classes generated adhere strictly to the schema specifications. This automation would significantly enhance code maintainability and reduce the potential for bugs when handling JSON data.

Enter the `JsonSerializableMeta` metaclass! When a class is being created, this metaclass swoops in to see if there's a schema attribute present. If it finds one, it calls on the `ObjectBuilder` from the `python_jsonschema_objects` package to craft classes that mirror the schema. These classes are then added as bases to the original class. This ensures that schema rules are enforced, simplifying validation and allowing for seamless JSON object serialization and deserialization.


```python
from abc import ABCMeta

from python_jsonschema_objects import ObjectBuilder
from python_jsonschema_objects.classbuilder import ProtocolBase

class JsonSerializableMeta(ABCMeta):
    def __new__(cls, name, bases, attrs) -> "JsonSerializableMeta":  # noqa: D102
        try:
            schema = attrs["schema"]
        except (KeyError, AttributeError):
            return super().__new__(cls, name, bases, attrs)

        builder = ObjectBuilder(schema)

        classes = builder.build_classes(
            strict=True,
            named_only=True,
            standardize_names=False,
        )

        json_schema = getattr(classes, name)

        return super().__new__(cls, name, (*bases, json_schema), attrs)

class JsonSerializable(ProtocolBase, metaclass=JsonSerializableMeta):
    pass
```

Let's now define a `User` class using the JSON schema for Spotify users.


```python
class User(JsonSerializable):
    schema = {
        "id": "https://dotify.com/track.user.json",
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "User",
        "type": "object",
        "required": ["display_name"],
        "properties": {
            "display_name": {"type": "string"},
            "external_urls": {
                "type": "object",
                "patternProperties": {"^.*$": {"type": "string"}},
                "minProperties": 1,
            },
            "href": {
                "type": "string",
                "pattern": "https://api.spotify.com/v1/users/.*",
            },
            "id": {"type": "string"},
            "type": {"type": "string", "pattern": "^user$"},
            "uri": {"type": "string", "format": "uri", "pattern": "spotify:user:.*"},
        },
    }
```

`ObjectBuilder` can take the schema directly or a path to a JSON schema file. So, the schema attribute at the class level might not always be a dictionary. Here's an example where we end up with the same User class, no matter which way we go:


```python
import json

schema = {
    "id": "https://dotify.com/track.user.json",
    "$schema": "http://json-schema.org/draft-04/schema#",
    "title": "User",
    "type": "object",
    "required": ["display_name"],
    "properties": {
        "display_name": {"type": "string"},
        "external_urls": {
            "type": "object",
            "patternProperties": {"^.*$": {"type": "string"}},
            "minProperties": 1,
        },
        "href": {
            "type": "string",
            "pattern": "https://api.spotify.com/v1/users/.*",
        },
        "id": {"type": "string"},
        "type": {"type": "string", "pattern": "^user$"},
        "uri": {"type": "string", "format": "uri", "pattern": "spotify:user:.*"},
    },
}

schema_path = "./schema.json"
with open(schema_path, "w") as file:
    file.write(json.dumps(schema, indent=4))


class User(JsonSerializable):
    schema = schema_path
```

- When you create a `User` object with all the required info, everything kicks off smoothly.
- But if you try to make a `User` without providing all the necessary details, it's like hitting a roadblock—validation errors pop up. This shows how the class sticks to its rules, making sure everything's in line with the schema during object creation.


```python
>>> User(display_name="Vassilis Sioros")
<User display_name=<Literal<str> Vassilis Sioros> external_urls=None href=None id=None type=None uri=None>
>>> User()
ValidationError: '['display_name']' are required attributes for User
```

It's like having a super-efficient assistant that automates all the boring class-building tasks for you. With metaclasses doing the heavy lifting, you can whip up complex class structures from JSON schemas with just a few lines of code. It's like getting a gourmet meal with minimal effort in the kitchen – metaclasses make complicated tasks seem like a breeze.

Alright, folks, that's a wrap for today's dive into Python metaclasses! I hope you've enjoyed uncovering their hidden powers alongside me. Until next time, keep exploring & keep coding!
