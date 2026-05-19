from arch_chat.registry import ARCH_REGISTRY, list_architectures


def test_registry_has_17_entries():
    assert len(ARCH_REGISTRY) == 17


def test_registry_unique_names():
    names = [e.name for e in list_architectures()]
    assert len(names) == len(set(names))


def test_registry_numbers_1_to_17():
    numbers = sorted(e.number for e in list_architectures())
    assert numbers == list(range(1, 18))
