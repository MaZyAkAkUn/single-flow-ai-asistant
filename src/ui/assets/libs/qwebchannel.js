"use strict";

var QWebChannel = function(transport, initCallback)
{
    if (typeof transport !== "object" || typeof transport.send !== "function") {
        console.error("The QWebChannel expects a transport object with a send function and onmessage callback property." +
                      " Given is: " + typeof transport);
        return;
    }

    var channel = this;
    this.transport = transport;

    this.send = function(data)
    {
        if (typeof data !== "string") {
            data = JSON.stringify(data);
        }
        channel.transport.send(data);
    }

    this.execCallbacks = {};
    this.execId = 0;
    this.objects = {};

    this.transport.onmessage = function(message)
    {
        var data = message.data;
        if (typeof data === "string") {
            data = JSON.parse(data);
        }
        switch (data.type) {
            case QWebChannel.signal:
                channel.handleSignal(data);
                break;
            case QWebChannel.response:
                channel.handleResponse(data);
                break;
            case QWebChannel.propertyUpdate:
                channel.handlePropertyUpdate(data);
                break;
            default:
                console.error("invalid message received:", message.data);
                break;
        }
    }

    this.exec = function(data, callback)
    {
        if (!callback) {
            // if no callback is given, send directly
            channel.send(data);
            return;
        }
        if (channel.execId === Number.MAX_VALUE) {
            // wrap
            channel.execId = 0;
        }
        if (data.hasOwnProperty("id")) {
            console.error("Cannot exec message with property id: " + JSON.stringify(data));
            return;
        }
        data.id = channel.execId++;
        channel.execCallbacks[data.id] = callback;
        channel.send(data);
    };

    this.handleSignal = function(message)
    {
        var object = channel.objects[message.object];
        if (object) {
            object.signalEmitted(message.signal, message.args);
        } else {
            console.warn("Unhandled signal: " + message.object + "::" + message.signal);
        }
    }

    this.handleResponse = function(message)
    {
        if (!message.hasOwnProperty("id")) {
            console.error("Invalid response message received: ", JSON.stringify(message));
            return;
        }
        channel.execCallbacks[message.id](message.data);
        delete channel.execCallbacks[message.id];
    }

    this.handlePropertyUpdate = function(message)
    {
        for (var i in message.data) {
            var data = message.data[i];
            var object = channel.objects[data.object];
            if (object) {
                object.propertyUpdate(data.signals, data.properties);
            } else {
                console.warn("Unhandled property update: " + data.object + "::" + data.signal);
            }
        }
        channel.execCallbacks[message.id](message.data);
        delete channel.execCallbacks[message.id];
    }

    this.debug = function(message)
    {
        channel.send({type: QWebChannel.debug, data: message});
    };

    channel.exec({type: QWebChannel.init}, function(data) {
        for (var objectName in data) {
            var object = new QObject(objectName, data[objectName], channel);
        }
        for (var objectName in data) {
            var object = channel.objects[objectName];
            object.unwrapProperties();
        }
        if (initCallback) {
            initCallback(channel);
        }
        channel.exec({type: QWebChannel.idle});
    });
};

QWebChannel.id = 0;
QWebChannel.signal = 1;
QWebChannel.propertyUpdate = 2;
QWebChannel.init = 3;
QWebChannel.idle = 4;
QWebChannel.debug = 5;
QWebChannel.invokeMethod = 6;
QWebChannel.connectToSignal = 7;
QWebChannel.disconnectFromSignal = 8;
QWebChannel.setProperty = 9;
QWebChannel.response = 10;

var QObject = function(name, data, webChannel)
{
    this.__id__ = name;
    webChannel.objects[name] = this;

    // List of keys of the property part of data sent in QWebChannel.init
    this.__propertyCache__ = [];

    var object = this;

    this.unwrapProperties = function()
    {
        for (var propertyIndex in object.__propertyCache__) {
            object.unwrapProperty(object.__propertyCache__[propertyIndex]);
        }
    }

    this.unwrapProperty = function(propertyName)
    {
        var propertyValue = object[propertyName];

        // send a property update and wait for the response to update variables
        object[propertyName] = function(newValue) {
            var args = [];
            for (var i = 0; i < arguments.length; i++) {
                args.push(arguments[i]);
            }
            webChannel.send({
                object: object.__id__,
                type: QWebChannel.setProperty,
                property: propertyName,
                value: args
            });
        }
    }

    this.signalEmitted = function(signalName, signalArgs)
    {
        var signal = object[signalName];
        signal.apply(object, signalArgs);
    }

    this.propertyUpdate = function(signals, propertyMap)
    {
        // update property cache
        for (var propertyIndex in propertyMap) {
            var propertyName = propertyMap[propertyIndex];
            if (object.hasOwnProperty(propertyName)) {
                // If it is already a property, it must be the initial value
                // Or a pending update
                // If it's a function, it's a writeable property, we don't want to overwrite the function
                // But we want to update the cache
                // Actually the init data contains current values.
                // We shouldn't overwrite the setter function though.
                continue;
             }
        }
        
    }

    for (var i in data.methods) {
        var methodName = data.methods[i][0];
        //var methodSignature = data.methods[i][1];
        object[methodName] = (function(methodName) {
            return function() {
                var args = [];
                for (var i = 0; i < arguments.length; i++) {
                    args.push(arguments[i]);
                }
                var callback = undefined;
                if (typeof args[args.length - 1] === "function") {
                    callback = args.pop();
                }
                webChannel.exec({
                    "type": QWebChannel.invokeMethod,
                    "object": object.__id__,
                    "method": methodName,
                    "args": args
                }, function(response) {
                    if (response !== undefined) {
                        var result = response;
                        if (callback) {
                            callback(result);
                        }
                    }
                });
            };
        })(methodName);
    }

    for (var i in data.properties) {
        var propertyName = data.properties[i][0];
        var propertyValue = data.properties[i][1];
        var propertyNotifySignal = data.properties[i][2];
        object[propertyName] = propertyValue;
        object.__propertyCache__.push(propertyName);
        
        if (propertyNotifySignal) {
             if (!object[propertyNotifySignal]) {
                  object[propertyNotifySignal] = {
                      connect: function(callback) {
                          this.__callbacks__.push(callback);
                      },
                      disconnect: function(callback) {
                             var idx = this.__callbacks__.indexOf(callback);
                             if (idx != -1) {
                                 this.__callbacks__.splice(idx, 1);
                             }
                      },
                      __callbacks__: []
                  };
                  object[propertyNotifySignal].apply = function(receiver, args) {
                      for (var i in this.__callbacks__) {
                          this.__callbacks__[i].apply(receiver, args);
                      }
                  };
             }
        }
    }

    for (var i in data.signals) {
        var signalName = data.signals[i][0];
        //var signalSignature = data.signals[i][1];
        object[signalName] = {
            connect: (function(signalName) {
                return function(callback) {
                    if (typeof callback !== "function") {
                        console.error("Bad callback given to connect to signal " + signalName);
                        return;
                    }
                    object[signalName].__callbacks__.push(callback);
                    // tell server we want to listen to this signal
                    webChannel.exec({
                        type: QWebChannel.connectToSignal,
                        object: object.__id__,
                        signal: signalName
                    });
                };
            })(signalName),
            disconnect: (function(signalName) {
                return function(callback) {
                    var idx = object[signalName].__callbacks__.indexOf(callback);
                    if (idx != -1) {
                        object[signalName].__callbacks__.splice(idx, 1);
                        // Tell server if no one is listening?
                        // Qt WebChannel doesn't seem to have a disconnectFromSignal message in this version usually,
                        // but logic suggests we should unsub if count is 0. 
                        // Simplified here.
                        webChannel.exec({
                            type: QWebChannel.disconnectFromSignal,
                            object: object.__id__,
                            signal: signalName
                        });
                    }
                };
            })(signalName),
            __callbacks__: []
        };
        object[signalName].apply = function(receiver, args) {
            for (var i in this.__callbacks__) {
                this.__callbacks__[i].apply(receiver, args);
            }
        };
    }
};
