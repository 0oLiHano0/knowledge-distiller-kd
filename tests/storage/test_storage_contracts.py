import inspect
import pytest
from typing import get_type_hints, Callable, Any

from knowledge_distiller_kd.storage.storage_interface import StorageInterface
from knowledge_distiller_kd.storage.orm_storage import ORMStorage

# TODO: Add other StorageInterface implementations here if needed in the future
IMPLEMENTATIONS_TO_TEST = [ORMStorage]

def get_public_methods(cls: type) -> dict[str, Callable]:
    """Helper function to get public methods of a class."""
    # Exclude magic methods and private methods
    methods = inspect.getmembers(cls, predicate=inspect.isfunction)
    return {name: func for name, func in methods if not name.startswith('_')}

@pytest.fixture(scope="module")
def interface_methods() -> dict[str, Callable]:
    """Fixture to provide the public methods of StorageInterface."""
    return get_public_methods(StorageInterface)

@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS_TO_TEST)
def test_implementation_methods_exist(implementation_cls: type, interface_methods: dict[str, Callable]):
    """Test that each method defined in the interface exists in the implementation."""
    implementation_methods = get_public_methods(implementation_cls)
    missing_methods = set(interface_methods.keys()) - set(implementation_methods.keys())
    assert not missing_methods, f"{implementation_cls.__name__} is missing methods defined in StorageInterface: {missing_methods}"

    extra_methods = set(implementation_methods.keys()) - set(interface_methods.keys())
    if extra_methods:
        print(f"Warning: {implementation_cls.__name__} has extra public methods not in StorageInterface: {extra_methods}") # Not strictly a failure, but good to know

@pytest.mark.parametrize("implementation_cls", IMPLEMENTATIONS_TO_TEST)
def test_method_signatures_match(implementation_cls: type, interface_methods: dict[str, Callable]):
    """Test that method signatures (parameters and return type) match the interface."""
    implementation_methods = get_public_methods(implementation_cls)

    for method_name, interface_method in interface_methods.items():
        if method_name not in implementation_methods:
            continue # Existence is checked in the previous test

        implementation_method = implementation_methods[method_name]

        # Get signatures
        try:
            interface_sig = inspect.signature(interface_method)
            implementation_sig = inspect.signature(implementation_method)
        except ValueError as e:
            pytest.fail(f"Could not get signature for method '{method_name}' in {implementation_cls.__name__} or StorageInterface: {e}")

        # Compare return types
        # Use get_type_hints to resolve forward references if any
        try:
            interface_return_type = get_type_hints(interface_method).get('return', inspect.Signature.empty)
            implementation_return_type = get_type_hints(implementation_method).get('return', inspect.Signature.empty)
        except Exception as e:
             pytest.fail(f"Error resolving type hints for method '{method_name}' in {implementation_cls.__name__} or StorageInterface: {e}")


        assert interface_return_type == implementation_return_type, \
            f"Method '{method_name}': Return type mismatch in {implementation_cls.__name__}. " \
            f"Interface: '{interface_return_type}', Implementation: '{implementation_return_type}'"

        # Compare parameters (name, order, type, default, kind)
        interface_params = interface_sig.parameters
        implementation_params = implementation_sig.parameters

        assert list(interface_params.keys()) == list(implementation_params.keys()), \
            f"Method '{method_name}': Parameter names/order mismatch in {implementation_cls.__name__}. " \
            f"Interface: {list(interface_params.keys())}, Implementation: {list(implementation_params.keys())}"

        for param_name in interface_params:
            interface_param = interface_params[param_name]
            implementation_param = implementation_params[param_name]

            # Compare parameter kind (POSITIONAL_OR_KEYWORD, VAR_POSITIONAL, etc.)
            assert interface_param.kind == implementation_param.kind, \
                f"Method '{method_name}', Parameter '{param_name}': Kind mismatch in {implementation_cls.__name__}. " \
                f"Interface: {interface_param.kind}, Implementation: {implementation_param.kind}"

            # Compare parameter type annotations
            # Use get_type_hints for the specific method to resolve types correctly within its scope
            try:
                interface_param_type = get_type_hints(interface_method).get(param_name, inspect.Parameter.empty)
                implementation_param_type = get_type_hints(implementation_method).get(param_name, inspect.Parameter.empty)
            except Exception as e:
                pytest.fail(f"Error resolving type hints for parameter '{param_name}' in method '{method_name}': {e}")


            assert interface_param_type == implementation_param_type, \
                f"Method '{method_name}', Parameter '{param_name}': Type annotation mismatch in {implementation_cls.__name__}. " \
                f"Interface: '{interface_param_type}', Implementation: '{implementation_param_type}'"

            # Compare parameter default values
            assert interface_param.default == implementation_param.default, \
                f"Method '{method_name}', Parameter '{param_name}': Default value mismatch in {implementation_cls.__name__}. " \
                f"Interface: {interface_param.default}, Implementation: {implementation_param.default}" 