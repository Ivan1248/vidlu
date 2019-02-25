def _scoped(*superclasses):
    class ScopedModuleExtension(*superclasses):
        # def __del__(self):
        #    for c in self.children():
        #        self._unregister_as_parent(c)

        def _register_as_parent(self, child_name, child_module):
            if hasattr(child_module, 'parents'):
                child_module.parents[self] = child_name
            else:
                child_module.parents = {self: child_name}

        def _unregister_as_parent(self, child_module):
            del child_module.parents[self]

        def __setattr__(self, key, value):
            super().__setattr__(key, value)
            # if isinstance(value, nn.Module):
            #    self._register_as_parent(key, value)

        def add_module(self, *args, **kwargs):
            super().add_module(name, module)
            self._register_as_parent(name, module)

        @property
        def scopei(self):
            if hasattr(self, 'parents'):
                scopes = [f'{par.scope}.{name}' for par, name in self.parents.items()]
                if len(scopes) == 1:
                    return scopes[0]
                return '(' + ', '.join(scopes) + ')'
            return 'ROOT'

        def get_parents(self):
            import gc
            for module in [r for r in gc.get_objects() if isinstance(r, (nn.Module))]:
                for k, v in module.named_children():
                    if v is self:
                        yield module, k
            """for r_modules in [r for r in gc.get_referrers(self)
                              if isinstance(r, (dict, OrderedDict))]:
                found = False
                for r__dict__ in [r for r in gc.get_referrers(r_modules)
                                  if type(r) is dict and r.get('_modules', None) is r_modules]:
                    for r_parent in gc.get_referrers(r__dict__):
                        if isinstance(r_parent, nn.Module) and r__dict__ is r_parent.__dict__:
                            yield r_parent, get_key(r_modules, self)
                            found = True
                            break
                    if found:
                        break"""

        @property
        def scope(self):
            scopes = [f'{par.scope}.{name}' for par, name in self.get_parents()]
            if len(scopes) == 1:
                return scopes[0]
            return '(' + ', '.join(scopes) + ')'

    return ScopedModuleExtension
